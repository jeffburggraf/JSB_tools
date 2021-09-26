"""
MaestroListFile class for reading and analysing Ortec Lis files.
Todo:
    Implement a method for efficiency calibration
"""
from __future__ import annotations
import plotly.graph_objects as go
import marshal
import struct
from struct import unpack, calcsize
from pathlib import Path
import datetime
from bitstring import BitStream
import numpy as np
from matplotlib import pyplot as plt
from datetime import timezone
from typing import List, Union, Tuple, Iterable
import filetime
from functools import cached_property
import time
from uncertainties.core import UFloat
from JSB_tools import ProgressReport, convolve_gauss, mpl_hist, calc_background, interpolated_median, shade_plot
from JSB_tools.spe_reader import SPEFile

# HERE = pytz.timezone('US/Mountain')
OLE_TIME_ZERO = datetime.datetime(1899, 12, 30, 0, 0, 0)
cwd = Path(__file__).parent


def ole2datetime(oledt):
    return OLE_TIME_ZERO + datetime.timedelta(days=float(oledt))


def get_spe_lines(_l: MaestroListFile):
    """
    Notes:
        n_adc_channels - 1: this is bc the number before the channel counts in the Maestro SPE file is one less than the
            number of channels since 0 is included as a channel.
    Args:
        _l:

    Returns:

    """
    datetime_str = _l.start_time.strftime(MaestroListFile.datetime_format)
    sample_des = _l.description
    if sample_des == '':
        sample_des = 'No sample description was entered.'

    spe_lines = ["$SPEC_ID:", sample_des, '$SPEC_REM:', f'DET# {_l.det_id_number}',
                 f'DETDESC# {_l.device_address}', 'AP# Maestro Version', '$DATE_MEA:', datetime_str, '$MEAS_TIM:',
                 f'{_l.livetimes[-1]:.4f} {_l.realtimes[-1]:.4f}', '$DATA:', f'0 {_l.n_adc_channels - 1}']
    for counts in _l.get_erg_spectrum()[0]:
        spe_lines.append(f'       {counts}')
    fmt = lambda x: f'{x:.6E}'
    erg_cal_line = ' '.join(map(lambda x: f'{x:.6f}', _l.erg_calibration[:2]))
    mca_cal_line = ' '.join(map(fmt, _l.erg_calibration))
    mca_cal_line += f' {_l.erg_units}'
    shape_cal_line = ' '.join(map(fmt, _l.shape_calibration))
    spe_lines.extend(map(str, ['$ROI:', '0', '$PRESETS:', 'None', '0', '0','$ENER_FIT:',
                         erg_cal_line, '$MCA_CAL:', len(_l.erg_calibration), mca_cal_line,
                         '$SHAPE_CAL:', len(_l.shape_calibration), shape_cal_line]))
    return spe_lines


class MaestroListFile:
    datetime_format = '%m/%d/%Y %H:%M:%S'
    file_datetime_format = '%Y-%m-%d_%H-%M-%S.%f--%Z%z'
    """
    Start time is determined by the Connections Win_64 time sent right after the header. This clock doesn't
    necessarily start with the 10ms rolling-over clock, so the accuracy of the start time is +/- 5ms.
    The SAMPLE READY and GATE counters are also +/- 5 ms.
    Relative times between ADC events are supposedly +/- 200ns. I will need to confirm this.
    
    The DSPEC50 appears to halt counting events when the SAMPLE_READY port is reading a TTL voltage. 
    
    Attributes:
        self.energies: numpy array of recorded energies (chronological)
        self.adc_values: numpy array of recorded ADC values (chronological)
        self.times: numpy array of recorded times
        
        self.__needs_updating__: A flag variable that tells code to re-evaluate self.energies (and similar), in case of 
            e.g. a change in the energy calibration.
        self._original_path: Path of .Lis file data was originally read from.
    """
    def __init__(self, path, max_words=None, debug=False):
        """
        self.adc_zero_time is the time you want to use for determining the system clock time of ADC events. DO NOT use
        the start time in the header (self.start_time)
        Args:
            path:
            max_words:
            debug:
        """
        path = Path(path)
        self._original_path = path
        self._spe: SPEFile  = None

        if not path.exists():  # assume file name given. Use JSB_tools/user_saved_data/SpecTestingData
            path = cwd/'user_saved_data'/'SpecTestingData'/path

        assert path.exists(), f'List file not found,"{path}"'

        with open(path, 'rb') as f:
            lst_header = self.read('i', f)

            self.adc_values: Union[np.ndarray, List[float]] = []
            assert lst_header == -13, f"Invalid list mode header. Should be '-13', got " \
                                      f"'{lst_header}' instead."
            self.list_style = self.read('i', f)
            if self.list_style != 2:
                raise NotImplementedError("Digibase and other list styles not yet implemented.")
            ole2datetime(self.read('d', f))  # read start time. Dont use this value.
            self.device_address = self.read("80s", f)
            self.MCB_type_string = self.read("9s", f)
            self.serial_number = self.read("16s", f)
            s = self.read('80s', f)
            self.description = s
            self.valid_erg_calibration = self.read('1s', f)
            self.erg_units = self.read("4s", f)
            self.erg_calibration = [self.read('f', f) for i in range(3)]
            self.valid_shape_calibration = self.read('1s', f)
            self.shape_calibration = [self.read('f', f) for i in range(3)]
            self.n_adc_channels = self.read('i', f)
            self.det_id_number = self.read('i', f)
            self.total_real_time = self.read('f', f)
            self.total_live_time = self.read('f', f)
            self.read('9s', f)
            self.start_time: Union[datetime.datetime, None] = None

            self.n_rollovers = 0  # Everytime the clock rolls over ()every 10 ms),
            # we see an RT word indicating the number of roll overs.
            self.wintime = [None] * 8

            self.livetimes = []
            self.realtimes = []
            self.sample_ready_state = []
            self.gate_state = []
            self.__10ms_counts__ = 0  # The number of ADC events each 10ms clock tick. Used for self.counts_per_sec.
            self.counts_per_sec = []

            self.times = []
            self.n_words = 0

            f_size = path.stat().st_size
            self.max_words = max_words
            prog = ProgressReport(f_size if self.max_words is None else self.max_words)

            t0 = time.time()
            while self.process_32(f, debug=debug):
                if self.n_words % 200 == 0:
                    if self.max_words is None:
                        progress_index = f.tell()
                    else:
                        progress_index = self.n_words

                    prog.log(progress_index)

                # print(progress_index, f_size)
                if self.max_words is not None and self.n_words > self.max_words:
                    if len(self.livetimes) == len(self.realtimes):
                        break
                self.n_words += 1
        print(f"Done reading list data. Rate: {int(len(self.times)/(time.time()-t0)):.2g} events read per second.")

        if time.time() - t0 > 25:
            _print_progress = True
            print("Processing...")
        else:
            _print_progress = False

        self.counts_per_sec = np.array(self.counts_per_sec)/10E-3

        if debug:
            print("Start time: ", self.start_time)
            print("Live time: ", self.total_live_time)
            print("Real time: ", self.total_real_time)
            print("Total % live: ", self.total_live_time/self.total_real_time)
            print("N events/s = ", len(self.times) / self.total_real_time)
            print(f"Total counts: {self.n_counts}")

        self.adc_values = np.array(self.adc_values)
        t_log = time.time()
        if _print_progress:
            print('Converting pulse height data to energy...', end='')
        self.energies = self.channel_to_erg(self.adc_values)
        if _print_progress:
            print(f'Done. ({time.time() - t_log:.1f} seconds)')
        if _print_progress:
            t_log = time.time()
            print("Done.\nCalculating livetime fractions...", end='')

        dead_time_corr_window = 10

        self.fraction_live = self._calc_fraction_live(self.livetimes, self.realtimes, dead_time_corr_window)

        if _print_progress:
            print(f'Done. ({time.time() - t_log:.1f} seconds)')

        if _print_progress:
            t_log = time.time()
            print("Converting data to numpy arrays...", end='')
        self.sample_ready_state = np.array(self.sample_ready_state)
        self.gate_state = np.array(self.gate_state)
        self.times = np.array(self.times)
        self.energies = np.array(self.energies)
        self.realtimes = np.array(self.realtimes)
        self.livetimes = np.array(self.livetimes)
        if _print_progress:
            print(f'Done. ({time.time() - t_log:.1f} seconds)')

        self.__needs_updating__ = False
        self._energy_binned_times = None

        if debug:
            plt.plot(self.realtimes, self.livetimes)
            plt.xlabel("Real-time [s]")
            plt.ylabel("Live-time [s]")
            plt.figure()

            plt.plot(self.realtimes, self.fraction_live)
            plt.title("")
            plt.xlabel("Real-time [s]")
            plt.ylabel("% live-time")

    def time_offset(self, t):
        """
        Offset all times by `t`. Useful for activation applications. For example, if the signal of interest begins at
            some point in time during data acquisition due to nuclear activation from a particle accelerator.
            This time may be determined using a pulse fed into the SAMPLE READY port. In this case, do the following:
                l = MaestroListFile(...)
                times = l.realtimes[np.where(l.sample_ready_state == 1)]
                time = np.median(times)
                l.time_offset(-time)
        Args:
            t: Time to offset in seconds.

        Returns: None

        """
        self.times = self.times + t

    @staticmethod
    def _calc_fraction_live(livetimes, realtimes, dead_time_corr_window=10):
        fraction_live = np.gradient(livetimes) / np.gradient(realtimes)
        fraction_live = convolve_gauss(fraction_live, dead_time_corr_window)

        # correct edge effects
        fraction_live[0:dead_time_corr_window // 2] = \
            np.median(fraction_live[dead_time_corr_window // 2:dead_time_corr_window])

        fraction_live[-dead_time_corr_window // 2:] = \
            np.median(fraction_live[-dead_time_corr_window: -dead_time_corr_window // 2])
        return fraction_live

    @property
    def __default_spe_path__(self):
        """
        The default path the MaestroList creates when converting List to Spe
        Returns:

        """
        return self._original_path.with_name(f'_{self._original_path.name}').with_suffix('.Spe')

    def list2spe(self, save_path: Union[Path, None] = None, write_file=False) -> SPEFile:
        """
        Generate and write to disk an ASCII Spe file form data in the Lis file.
        Produces files identical to (and readable by) the Maestro application.
        Args:
            save_path: Path for new file. If None, use original file but with .Spe for the suffix.
            write_file: If True, write the SPE file to disk

        Returns: SPE object.

        """
        if save_path is None:
            save_path = self.__default_spe_path__
        else:
            save_path = Path(save_path)
        save_path = save_path.with_suffix(".Spe")
        spe_text = '\n'.join(get_spe_lines(self))
        if write_file:
            with open(save_path, 'w') as f:
                f.write(spe_text)
        spe = SPEFile(save_path)
        assert all(self.erg_bins == spe.erg_bins)
        assert all(self.get_erg_spectrum() == spe.counts)
        return spe

    def read(self, byte_format, f, debug=False):
        s = calcsize(byte_format)
        _bytes = f.read(s)
        out = unpack(byte_format, _bytes)
        if debug:
            print(f"Byte format: {byte_format}")
            print(f"Calc'd size: {s}")
            print(f"out: {out}")
        if len(out) == 1:
            if byte_format[-1] == 's':
                return out[0].rstrip(b'\x00').decode("utf-8", "strict")
            return out[0]
        else:

            return out

    @cached_property
    def SPE(self) -> SPEFile:
        """
        Get SPE object from Lis data.
        Returns:

        """
        p = self.__default_spe_path__
        if self._spe is not None:
            return self._spe
        else:
            if p.exists():
                out = SPEFile(p)
            else:
                out = self.list2spe()
        self._spe = out

    def set_energy_cal(self, *coeffs):
        self.erg_calibration = np.array(coeffs)
        self.energies = self.channel_to_erg(self.adc_values)
        self.SPE.set_energy_cal(coeffs)
        self.__needs_updating__ = True

    def read_32(self, f) -> str:
        """
        Read a 32-bit word and convert to a string of 0s and 1s.
        Returns:

        """
        out = f"{self.read('I', f):032b}"
        return out

    def process_32(self, f, debug=False):
        """
        Read and store a 32-bit word.

        About the words:
        - LiveTime word: a clock that increments every 10ms of live time (relevant to deadtime calculation).

        - RealTime word: a clock that increments every 10ms of real time.

        - ADC word: contains a 14 binary integer representing an ADC channel, and a 16 bit binary integer representing
        the number of Fast Clock ticks since the last FastClock reset. The FastClock value increments every 200ns, and
        rolls over every 10 ms, or 50000 ticks (thus, the RealTime word is equal to the number of times the FastClock
        has rolled over). Thus, the real (actual) time of any event relative to the start of acquisition is:
                    (RealTime value) * [10ms] + (FastClock value) * [200ns])

        - Win64 Time: a series of three 32-bit words representing system time. This is the very first word after the
         header, thus is used to get the system time for the beginning of acquisition.

        - Hardware Time: Occasionally this word appears apparently for no reason. It comes from the same time sourse
        as the 200ns ADC clock. I haven't found a use for this yet.

        - ADC CRM: Appears every 10ms of real time. Represents the number of counts during the previous 10ms period.
        This generally an over-estimate of the actual count rate since rejected pulses are also counted.

        - Counter A: State of the SAMPLE READY port. Appears every 10ms.

        - Counter B: State of the GATE port. Appears every 10ms.

        Args:
            f: IO binary file.
            debug: Print information about each 32-bit word.

        Returns: None

        """
        try:
            word = self.read_32(f)
        except struct.error:
            return False
        word_debug = None
        if word[:2] == "11":  # ADC word
            adc_value = int(word[2:16], 2)
            self.adc_values.append(adc_value)
            n_200ns_ticks = int(word[16:], 2)
            adc_time = self.n_rollovers * 10E-3 + n_200ns_ticks * 200E-9
            self.times.append(adc_time)
            self.__10ms_counts__ += 1
            if debug:
                word_debug = f"ADC word, Value = {adc_value}, n_200ns_ticks = {n_200ns_ticks}, real time: {adc_time}," \
                             f"n rollovers: {self.n_rollovers}"

        elif word[:2] == "01":  # LiveTime word
            live_time_10ms = int(word[2:], 2)
            self.livetimes.append(10E-3*live_time_10ms)
            self.counts_per_sec.append(self.__10ms_counts__)
            self.__10ms_counts__ = 0

            if debug:
                word_debug = f"LT word: {live_time_10ms}"

        elif word[:2] == "10":  # RealTime word
            real_time_10ms = int(word[2:], 2)
            self.realtimes.append(10E-3*real_time_10ms)
            self.n_rollovers = real_time_10ms
            if debug:
                word_debug = f"RT word: {real_time_10ms}"

        elif word[:8] == '00000000':
            hrd_time = int(word[16:], 2)
            if debug:
                word_debug = f"Hdw Time: {hrd_time}"

        elif word[:8] == '00000100':
            adc_crm = int(word[16:], 2)
            # self.n_rollovers += 1
            if debug:
                word_debug = f"ADC CRM: {adc_crm}"

        elif word[:8] == '00000101':
            sample_ready = int(word[16:], 2)
            self.sample_ready_state.append(sample_ready)
            if debug:
                word_debug = f"Counter A (Sample Ready): {sample_ready}"

        elif word[:8] == '00000110':
            gate = int(word[16:], 2)
            self.gate_state.append(gate)

            if debug:
                word_debug = f"Counter B (Gate): {gate}"

        elif word[:8] == '00000001':  # Window time incoming
            word1 = word
            word2 = self.read_32(f)
            word3 = self.read_32(f)
            wintime = [0, 0, 0, 0, 0, 0, 0, 0]
            wintime[7] = word1[-8:]
            wintime[6] = word1[-16:-8]
            wintime[5] = word1[8:16]
            wintime[4] = word2[-8:]
            wintime[3] = word2[-16:-8]
            wintime[2] = word2[8:16]
            wintime[1] = word3[-8:]
            wintime[0] = word3[-16:-8]
            w_time = "".join(wintime)
            w_time = BitStream(bin=w_time).unpack('uintbe:64')[0]
            w_time = filetime.to_datetime(w_time)
            w_time = w_time.replace(tzinfo=timezone.utc).astimezone(tz=None)

            if self.start_time is None:  # only set this for the first time we see a Win64 time.
                self.start_time = w_time

            if debug:
                word_debug = f"WinTime: {w_time.isoformat()}"
        else:
            if debug:
                word_debug = f"Unknown word: {word}"

        if debug:
            print(f"{self.n_words} {word_debug}")
        return True

    def header_info(self):
        s = f"List style: {self.list_style}\n"
        s += f"start time: {self.start_time.isoformat()}\n"
        s += f"Device address: {self.device_address}\n"
        s += f"MCB type: {self.MCB_type_string}\n"
        s += f"Device serial number: {self.serial_number}\n"
        s += f"Sample description: {self.description}\n"
        s += f"valid_erg_calibration?: {bool(self.valid_erg_calibration)}\n"
        s += f"Energy units: {self.erg_units}\n"
        s += f"Energy calibration: {self.erg_calibration}\n"
        s += f"valid_shape_calibration?: {bool(self.valid_shape_calibration)}\n"
        s += f"Shape calibration: {self.shape_calibration}\n"
        s += f"# of adc channels: {self.n_adc_channels}\n"
        s += f"detector ID: {self.det_id_number}\n"
        s += f"Maestro real time: {self.total_real_time}\n"
        s += f"Maestro live time: {self.total_live_time}\n"
        return s

    @property
    def n_counts(self):
        return len(self.times)

    @cached_property
    def sample_ready_median(self):
        """
        median of the times for which SAMPLE READY port is ON.
        For when SAMPLE READY port is used as a time stamp.
        Returns:

        """
        return np.median(self.realtimes[np.where(self.sample_ready_state == 1)])

    def get_time_dependence(self, energy, bins: Union[str, int, np.ndarray] = 'auto', signal_window_kev: float = 3,
                            bg_window_kev=None,
                            bg_offsets: Union[None, Tuple, float] = None, make_rate=False,
                            offset_sample_ready=False,
                            debug_plot=False):
        """
        Get the time dependence around erg +/- signal_window_kev/2. Baseline is estimated and subtracted.
        Estimation of baseline is done by taking the median rate ( with an energy of `energy` +/- `bg_window_kev`/2,
        excluding signal window) of events in each time bin. The median is then normalized to the width of the signal
        window (in KeV), and finally subtracted from the time dependence of events in the signal window.
        Args:
            energy:
            bins:
            signal_window_kev: Width around `energy` that will be considered the signal.
            bg_window_kev: Size of the window used for baseline estimation.
            bg_offsets: Offset the baseline window from the center (default will avoid signal window by distance of
                half `signal_window_kev`)
            make_rate: Makes units in Hz instead of counts.
            offset_sample_ready: Subtract the median of the times for which SAMPLE READY port is ON.
            debug_plot:


        Returns: Tuple[signal window rate, baseline rate estimation, bins used]
        """
        if bg_offsets is None:
            bg_offsets = [2*signal_window_kev]*2
        elif isinstance(bg_offsets, Iterable):
            bg_offsets = tuple(map(lambda x: 2*signal_window_kev if x is None else x, bg_offsets))
            if not len(bg_offsets) == 2:
                raise ValueError('Too many elements in arg `bg_offsets')
        elif isinstance(bg_offsets, (float, int)):
            bg_offsets = tuple([bg_offsets]*2)
        else:
            raise ValueError(f'Bad value for `bg_offsets`: {bg_offsets}')

        if bg_window_kev is None:
            bg_window_kev = 6*signal_window_kev

        bg_offsets = np.abs(bg_offsets)

        bg_window_left_bounds = np.array([-bg_window_kev/2, 0]) + energy - bg_offsets[0]
        bg_window_right_bounds = np.array([0, bg_window_kev/2]) + energy + bg_offsets[1]
        sig_window_bounds = [energy-signal_window_kev/2, energy+signal_window_kev/2]

        bg_window_left_is = self.__erg_index__(bg_window_left_bounds)
        bg_window_right_is = self.__erg_index__(bg_window_right_bounds)

        bg_times = self.__energy_binned_times__[slice(*bg_window_left_is)]
        bg_times += self.__energy_binned_times__[slice(*bg_window_right_is)]

        sig_times, n_sig_erg_bins = self.energy_slice(*sig_window_bounds, return_num_bins=True)

        if isinstance(bins, str):
            _, time_bins = np.histogram(sig_times, bins=bins)
        elif isinstance(bins, int):
            time_bins = np.linspace(sig_times[0], sig_times[-1], bins)
        else:
            assert hasattr(bins, '__iter__')
            time_bins = bins

        bg = np.array([interpolated_median(x) for x in np.array([np.histogram(times, bins=time_bins)[0]for times in bg_times]).transpose()])
        bg *= n_sig_erg_bins

        sig = np.histogram(sig_times, bins=time_bins)[0]

        if debug_plot:
            # fig, ax = plt.subplots()
            for index, (b1, b2) in enumerate(zip(time_bins[:-1], time_bins[1:])):
                ax, y = self.plot_erg_spectrum(bg_window_left_bounds[0]-signal_window_kev,
                                               bg_window_right_bounds[-1] + signal_window_kev, b1, b2,
                                               return_bin_values=True)
                ax.plot(bg_window_left_bounds, [bg[index]/n_sig_erg_bins] * 2,
                        label='Baseline est.', color='red', ls='--')
                ax.plot(bg_window_right_bounds, [bg[index] / n_sig_erg_bins] * 2, color='red', ls='--')
                ax.plot(sig_window_bounds, [sig[index]/n_sig_erg_bins] * 2,
                        label='Sig. +bg. est.', ls='--')
                ax.legend()
                shade_plot(ax, sig_window_bounds, label='Signal window')
                shade_plot(ax, bg_window_left_bounds, color='red', label='Bg. window')
                shade_plot(ax, bg_window_right_bounds, color='red')
                plt.show()

        if make_rate:
            b_widths = time_bins[1:] - time_bins[:-1]
            sig /= b_widths
            bg /= b_widths
        if offset_sample_ready:
            time_bins -= self.sample_ready_median

        return (sig - bg), bg, time_bins

    def est_half_life(self, energy, window: float = 1.5, bg_window_width=None,
                      left_right_bg_offset:
                      Union[None, Tuple[Union[float, int, None], Union[float, int, None]], float] = None,
                      rtol=1E-5,
                      debug_plot=False):
        """

        Args:
            energy:
            window:
            bg_window_width: Window for bg estimation. Should be large-ish. If None, 10*window. Data in the energy range
                of twice the signal window are not included in background estimation.
            left_right_bg_offset:
                Distance of bg offset(s) from energy center. By default, use value of `window` for left and right.
                    - If tuple of len 2, use separate offsets for left and right bs windows. If None appears in a tuple
                      entry, don't include background window for that side.
                    - If a float, use same value for both left and right offsets.
            rtol: Smaller = better convergence. 
            debug_plot:

        Returns:

        """
        if bg_window_width is None:
            bg_window_width = 20*window

        if left_right_bg_offset is None:
            left_right_bg_offset = (window, window)
        elif isinstance(left_right_bg_offset, Iterable):
            left_right_bg_offset = list(left_right_bg_offset)
            assert len(left_right_bg_offset) == 2, '`left_right_bg_offset` must be of length 2 when supplying ' \
                                                   'asymmetrical windows.'
            left_right_bg_offset = tuple(map(lambda x: window if x is None else x, left_right_bg_offset))
        else:
            assert isinstance(left_right_bg_offset, (int, float)),'Invalid type of `left_right_bg_offset` ' \
                                                                  '(must be number)'
            left_right_bg_offset = tuple([left_right_bg_offset]*2)

        sig_window = np.array([-window/2, window/2]) + energy
        times_raw, n_bins_center = self.energy_slice(*sig_window, return_num_bins=True)

        bg_window_left = np.array([-bg_window_width/2, 0]) + energy - left_right_bg_offset[0]
        bg_window_right = np.array([0, bg_window_width/2]) + energy + left_right_bg_offset[1]

        times_bg_left, n_bins_bg_left = self.energy_slice(*bg_window_left, return_num_bins=True)
        times_bg_right, n_bins_bg_right = self.energy_slice(*bg_window_right, return_num_bins=True)
        times_bg = np.concatenate([times_bg_right, times_bg_left])

        y_raw, bins = np.histogram(times_raw, bins='auto')
        y_bg, _ = np.histogram(times_bg, bins=bins)

        bin_centers = 0.5*(bins[:-1] + bins[1:])
        bg_scale = 1.0/(n_bins_bg_left+n_bins_bg_right)*n_bins_center
        y_bg = y_bg*bg_scale
        y_sig = y_raw - y_bg
        if debug_plot:
            ax = mpl_hist(bins, y_raw, title=f"tot., raw, and bg. for peak at {energy} +/- {window/2}", label='tot.')
            mpl_hist(bins, y_bg, ax=ax, label='bg')
            mpl_hist(bins, y_sig, ax=ax, label='sig')

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Counts')
            ax.legend()
            ax = self.plot_erg_spectrum(erg_min=energy-bg_window_width/1.5-window, erg_max=energy+bg_window_width/1.5+window,
                                   )
            # fig = plt.figure()
            # fig.suptitle(self.file_name)
            ax = plt.gca()
            y1, y2 = [ax.get_ylim()[0]]*2, [ax.get_ylim()[1]]*2
            ax.fill_between(bg_window_left, y1, y2, color='red', alpha=0.6,label='bg')
            ax.fill_between(bg_window_right, y1, y2, color='red', alpha=0.6)
            ax.fill_between(sig_window, y1, y2, color='blue', alpha=0.6, label='sig')
            ax.legend()
        min_time = bin_centers[np.argmax(y_sig)]
        print('min time:', min_time)

        times_raw = times_raw[np.where(times_raw >= min_time)] - min_time
        times_bg = times_bg[np.where(times_bg >= min_time)] - min_time
        weights = np.interp(times_raw + min_time, bin_centers, np.sqrt(y_raw))
        # weights_bg = np.interp(times_bg + min_time, bin_centers, np.sqrt(y_bg/bg_scale))
        # lambda_est = (len(times_raw)-len(times_bg))/(np.sum(times_raw) - np.sum(times_bg))
        data_len = len(times_raw)-bg_scale*len(times_bg)
        # data_len = np.sum(weights)-bg_scale*np.sum(weights)
        data_sum = np.sum(times_raw) - bg_scale*np.sum(times_bg)
        tmax = self.times[-1]

        def iterate(prev_lambda=None):
            # global i
            # i += 1
            if prev_lambda is None:
                out = data_len / data_sum
                print('init:', np.log(2)/out)
                return iterate(out)
            else:
                corr = -data_len * (tmax / (1 - np.e ** (tmax * prev_lambda)))
                # print(corr, data_len, data_sum)
                next_lambda = data_len / (data_sum + corr)
                # print(np.log(2)/next_lambda)
                if np.isclose(prev_lambda, next_lambda, rtol=rtol):
                    return next_lambda
                else:
                    return iterate(next_lambda)

        return np.log(2)/iterate()

    @property
    def file_name(self):
        return self._original_path.name

    def plot_percent_live(self, ax=None, **ax_kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.set_title(self._original_path.name)
        ax.plot(self.realtimes, self.fraction_live, **ax_kwargs)
        ax.set_xlabel("Real time [s]")
        ax.set_ylabel("Fraction livetime")
        return ax

    def get_deadtime_corr(self, t):
        i = np.searchsorted(self.livetimes, t)
        return self.deadtime_corrs[i]

    @cached_property
    def deadtime_corrs(self):
        indicies = np.searchsorted(self.realtimes, self.times)
        return 1.0/self.fraction_live[indicies]

    def channel_to_erg(self, channel) -> np.ndarray:
        return np.sum([channel ** i * c for i, c in enumerate(self.erg_calibration)], axis=0)

    @cached_property
    def erg_centers(self):
        return np.array([(b0+b1)/2 for b0, b1 in zip(self.erg_bins[:-1], self.erg_bins[1:])])

    @cached_property
    def erg_bins(self):
        channel_bins = np.arange(self.n_adc_channels + 1) - 0.5   #
        return self.channel_to_erg(channel_bins)

    def plot_count_rate(self, ax=None, smooth=None, **ax_kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if 'ds' not in ax_kwargs:
            ax_kwargs['ds'] = 'steps-post'
        if smooth is not None:
            assert smooth > 0 and isinstance(smooth, int)
            smooth = int(smooth)
            y = convolve_gauss(self.counts_per_sec, smooth)
        else:
            y = self.counts_per_sec
        ax.plot(self.realtimes, y, **ax_kwargs)
        ax.set_title(self._original_path.name)
        ax.set_xlabel("Real time [s]")
        ax.set_ylabel("Count rate [Hz]")
        return ax

    def plot_sample_ready(self, ax=None, **ax_kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if 'label' not in ax_kwargs:
            ax_kwargs['label'] = "SAMPLEREADY"
        ax.set_xlabel("Real time [s]")
        ax.set_ylabel("State")
        ax.plot(self.realtimes, self.sample_ready_state, **ax_kwargs)
        return ax

    def plot_gate(self, ax=None, **ax_kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if 'label' not in ax_kwargs:
            ax_kwargs['label'] = "GATE"
        ax.set_xlabel("Real time [s]")
        ax.set_ylabel("State")
        ax.plot(self.realtimes, self.gate_state, **ax_kwargs)
        return ax

    @staticmethod
    def __get_auto_data_dir():
        directory = cwd / 'user_saved_data' / 'list_data'
        if not directory.exists():
            directory.mkdir()
        return directory

    def __erg_index__(self, energies):
        """
        Get the index which corresponds to the correct energy bin(s).

        Examples:
            The result can be used to find the number of counts in the bin for 511 KeV:
                self.counts[self.erg_bin_index(511)]

        Args:
            energies:

        Returns:

        """
        out = np.searchsorted(self.erg_bins, energies, side='right') - 1

        return out

    @property
    def __energy_binned_times__(self) -> List[np.ndarray]:
        """
        The times of all events are segregated into the available energy bins.
        Returns:

        """
        if self.__needs_updating__ or self._energy_binned_times is None:
            self._energy_binned_times = [[] for _ in range(self.n_adc_channels)]
            erg_indicies = self.__erg_index__(self.energies)
            for i, t in zip(erg_indicies, self.times):
                self._energy_binned_times[i].append(t)
            self._energy_binned_times = [np.array(ts) for ts in self._energy_binned_times]
            self.__needs_updating__ = False
            return self._energy_binned_times
        else:
            return self._energy_binned_times

    def erg_bins_cut(self, erg_min, erg_max):
        """
        Get array of energy bins in range specified by arguments.
        Args:
            erg_min:
            erg_max:

        Returns:

        """
        i0 = self.__erg_index__(erg_min)
        i1 = self.__erg_index__(erg_max) + 1
        return self.erg_bins[i0: i1]
        # return self.erg_bins[np.where((self.erg_bins >= erg_min) & (self.erg_bins <= erg_max))]

    def get_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = None,
                         time_max: float = None, return_bins=False):
        """
        Get energy spectrum according to provided condition.
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            return_bins: Return the energy bins, e.g. for use in a histogram.

        Returns:

        """
        if erg_min is None:
            erg_min = self.erg_bins[0]
        if erg_max is None:
            erg_max = self.erg_bins[-1]

        def get_n_events():
            b = (time_min is not None, time_max is not None)
            if b == (0, 0):
                return len
            elif b == (1, 0):
                return lambda x: len(x) - np.searchsorted(x, time_min, side='left')
            elif b == (0, 1):
                return lambda x: np.searchsorted(x, time_max, side='right')
            else:
                return lambda x: np.searchsorted(x, time_max, side='right') - np.searchsorted(x, time_min, side='left')

        time_arrays = self.__energy_binned_times__[self.__erg_index__(erg_min): self.__erg_index__(erg_max)]
        func = get_n_events()
        out = np.fromiter(map(func, time_arrays), dtype=int)

        if return_bins:
            return out, self.erg_bins_cut(erg_min, erg_max)
        else:
            return out

    def plot_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = 0,
                          time_max: float = None, remove_baseline=False, title=None, ax=None, label=None,
                          return_bin_values=False):
        """
        Plot energy spectrum with time and energy cuts.
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            remove_baseline: If True, remove baseline.
            title:
            ax:
            label:
            return_bin_values:

        Returns:

        """
        if ax is None:
            fig = plt.figure()
            fig.suptitle(Path(self.file_name).name)
            ax = plt.gca()

        bin_values, bins = self.get_erg_spectrum(erg_min, erg_max, time_min, time_max, return_bins=True)
        if remove_baseline:
            bl = calc_background(bin_values)
            bin_values = -bl + bin_values
        mpl_hist(bins, bin_values, np.sqrt(bin_values), ax=ax, label=label)
        if title is None:
            if not (time_min == 0 and time_max is None):
                ax.set_title(f"{time_min:.1f} <= t <= {time_max:.1f}")
        else:
            ax.set_title(title)

        ax.set_xlabel('Energy [KeV]')
        ax.set_ylabel('Counts')
        if not return_bin_values:
            return ax
        else:
            return ax, bin_values

    def slicer(self, erg_min=None, erg_max=None, time_min=None, time_max=None):
        """
        Return an array that can be used to select the desired times or energies.
        Examples:
            s = l.slicer(erg_max=200, erg_min=50, time_max=34)
            plt.histogram(l.times[s])  # plot selected times
            plt.histogram(l.energies[s])  # plot selected energies
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:

        Returns:

        """
        if erg_min is not None:
            mask = (erg_min <= self.energies)
        else:
            mask = np.ones(len(self.times), dtype=bool)
        if erg_max is not None:
            mask &= (self.energies <= erg_max)

        if time_min is not None:
            mask[:np.searchsorted(self.times, time_min)] = False

        if time_max is not None:
            mask[np.searchsorted(self.times, time_max, side='right'):] = False

        return np.where(mask)

    def plotly(self, erg_min=None, erg_max=None, erg_bins='auto', time_bin_width=15,
               time_step: int = 5, percent_of_max=False):
        """
        Args:
            erg_min:
            erg_max:
            erg_bins:
            time_bin_width: Width of time range for each slider step.
            time_step: delta t between each slider step
            percent_of_max: It True, each bin is fraction of max.

        Returns:

        """
        if erg_min is None:
            erg_min = self.erg_bins[0]
        if erg_max is None:
            erg_max = self.erg_bins[-1]

        tmin = self.times[0]
        tmax = self.times[-1]
        time_centers = np.linspace(tmin, tmax, int((tmax-tmin) // time_step + 1))
        time_groups = [(max([tmin, b - time_bin_width / 2]), min([tmax, b + time_bin_width / 2])) for b in time_centers]
        time_bin_widths = [b1-b0 for b0, b1 in time_groups]

        energy_bins = self.erg_bins_cut(erg_min, erg_max)
        energy_bin_centers = (energy_bins[1:] + energy_bins[:-1])/2
        labels4bins = []

        ys = []

        for b_width, (b0, b1) in zip(time_bin_widths, time_groups):
            ys.append(self.get_erg_spectrum(erg_min, erg_max, b0, b1) / b_width)
            labels4bins.append((b0, b1))
        ys = np.array(ys)

        fig = go.Figure()

        y_tot = self.get_erg_spectrum(erg_min, erg_max)/(tmax-tmin)
        # y_tot *= np.max(ys)/np.max(y_tot)
        fig.add_trace(
            go.Bar(
                visible=True,
                name=f"All time",
                x=energy_bin_centers,
                y=y_tot,
                marker_color='red',
            )
        )

        steps = [dict(
                method="update",
                args=[{"visible": [True] + [False] * len(time_centers)},
                      {"title": f"All time"}],  # layout attribute
            )]

        _max_y = 0
        if percent_of_max:
            ys /= np.array([np.max(convolve_gauss(_y, 3)) for _y in ys])[:, np.newaxis]
        for index, (y, (t0, t1)) in enumerate(zip(ys, time_groups)):
            b_width = t1 - t0
            b_center = (t1+t0)/2
            if max(y)>_max_y:
                _max_y = max(y)

            label = f"t ~= {b_center:.1f} [s] ({t0:.1f} < t < {t1:.1f})"
            assert len(energy_bin_centers) == len(y), [len(energy_bin_centers), len(y)]
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    x=energy_bin_centers,
                    y=y,
                    marker_color='blue',
                    line={'shape': 'hvh'}
                ),
            )
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    x=energy_bin_centers,
                    y=convolve_gauss(y, 3),
                    ))
            step = dict(
                method="update",
                args=[{"visible": [False] * (2*len(time_centers) + 1)},
                      {"title": label}],  # layout attribute
            )
            step["args"][0]["visible"][2*index+1] = True  # Toggle trace to "visible"
            step["args"][0]["visible"][2*index+2] = True  # Toggle trace to "visible"
            steps.append(step)
        fig.update_yaxes(range=[0, _max_y*1.1])
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "time "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders, bargap=0, bargroupgap=0.0
        )
        fig.show()
        return fig

    def time_slice(self, time_min, time_max):
        """
        Return the energies of all events which occurred between time_min and time_max
        Args:
            time_min:
            time_max:

        Returns:

        """
        return self.energies[np.where((self.times >= time_min) & (self.times <= time_max))]

    def energy_slice(self, erg_min, erg_max, return_num_bins=False):
        """
        Return the times of all events with energy greater than `erg_min` and less than `erg_max`
        Args:
            erg_min:
            erg_max:
            return_num_bins: Whether to return the number of energy bins accessed

        Returns:

        """
        i0 = self.__erg_index__(erg_min)
        i1 = self.__erg_index__(erg_max)+1
        out = np.concatenate(self.__energy_binned_times__[i0: i1])
        if return_num_bins:
            return out, i1-i0
        else:
            return out

    def pickle(self, f_path=None):
        d = {'times': self.times,
             'max_words': self.max_words,
             'realtimes': self.realtimes,
             'livetimes': self.livetimes,
             'n_adc_channels': self.n_adc_channels,
             'fraction_live': self.fraction_live,
             'erg_calibration': self.erg_calibration,
             'gate_state': self.gate_state,
             'sample_ready_state': self.sample_ready_state,
             'adc_values': self.adc_values,
             'counts_per_sec': self.counts_per_sec,
             '_original_path': str(self._original_path),
             'start_time': datetime.datetime.strftime(self.start_time, MaestroListFile.datetime_format),
             'device_address': self.device_address,
             'list_style': self.list_style,
             'description': self.description,
             'valid_erg_calibration': self.valid_erg_calibration,
             'serial_number': self.serial_number,
             'MCB_type_string': self.MCB_type_string,
             'erg_units': self.erg_units,
             'shape_calibration': self.shape_calibration,
             'det_id_number': self.det_id_number,
             'total_real_time': self.total_real_time,
             'total_live_time': self.total_live_time,
             }
        d_np_types = {'gate_state': 'int', 'sample_ready_state': 'int', 'adc_values': 'int'}
        if f_path is None:
            f_path = self._original_path.parent / self.file_name
        f_path = Path(f_path).with_suffix('.marshal')
        with open(f_path, 'wb') as f:
            marshal.dump(d, f)
            marshal.dump(d_np_types, f)

    @classmethod
    def from_pickle(cls, fpath) -> MaestroListFile:

        fpath = Path(fpath).with_suffix('.marshal')

        if not fpath.exists():
            raise FileNotFoundError(f'MaestroListFile marshal path does not exist, "{fpath}"')
        with open(fpath, 'rb') as f:
            d = marshal.load(f)
            d_np_types = marshal.load(f)

        self = cls.__new__(cls)
        for k, v in d.items():
            if isinstance(v, bytes):
                if k in d_np_types:
                    t = eval(d_np_types[k])
                else:
                    t = float
                v = np.frombuffer(v, dtype=t)
            setattr(self, k, v)
        self._original_path = Path(self._original_path)
        self.start_time = datetime.datetime.strptime(self.start_time, MaestroListFile.datetime_format)
        self.__needs_updating__ = False
        self.energies = self.channel_to_erg(self.adc_values)
        self._energy_binned_times = None
        return self

    def __len__(self):
        return len(self.times)


if __name__ == '__main__':
    # a = []
    # for i in range(16000):
    #     a.append(list(np.random.randn(np.random.randint(200, 3000))))
    #
    # for i in range(4):
    #     t0 = time.time()
    #     b = np.array(a, dtype=object)
    #     print(time.time() - t0)
    l = MaestroListFile.from_pickle('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot132.Lis')
    # l = MaestroListFile(r'C:\Users\garag\PycharmProjects\IACExperiment\exp_data\Friday\shot122.Lis')
    # bg_spe = SPEFile(r'C:\Users\garag\PycharmProjects\IACExperiment\exp_data\tuesday\BG.Spe')
    # bg = bg_spe
    # l.plot_erg_spectrum()
    # print(l.counts)
    # l.pickle()

    # l.slicer()
    f = l.plotly(erg_min=40, erg_max=500, time_bin_width=20, time_step=2)
    f = l.plotly(erg_min=50, erg_max=500, time_bin_width=20, time_step=2, fig2=f)

    # l.get_time_dependence(218, debug_plot=True)
    plt.show()
    # l.plot_erg_spectrum()

    # sig, bg, bins = l.get_time_dependence(218.8, bins=15, signal_window_kev=3, bg_window_kev=30, debug_plot=True)
    # ax = mpl_hist(bins, sig, label='sig')
    # mpl_hist(bins, bg, label='sbg', ax=ax)

    # l.pickle()
    # l2 = MaestroListFile.from_pickle('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot132.Lis')
    # l2.pickle()
    # t0 = time.time()
    # hl = l.est_half_life(218.5, window=1.3, debug_plot=True, bg_window_width=3.5, left_right_bg_offset=(None, 2))
    # print(time.time() - t0)
    # print("Estimated half life:", hl)
    # # l.plot_erg_spectrum(remove_baseline=True)
    # # l2.plot_erg_spectrum(remove_baseline=True)


