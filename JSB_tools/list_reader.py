"""
MaestroListFile class for reading and analysing Ortec Lis files.
Todo:
    Implement a method for efficiency calibration
"""
from __future__ import annotations
import warnings
import plotly.graph_objects as go
import marshal
import struct
import pickle
from struct import unpack, calcsize
from pathlib import Path
import datetime
from bitstring import BitStream
import numpy as np
from matplotlib import pyplot as plt
from datetime import timezone
from typing import List, Union, Tuple, Iterable, Callable, Dict
import filetime
from functools import cached_property
import time
from uncertainties.core import UFloat, ufloat
from uncertainties import unumpy as unp
from JSB_tools import ProgressReport, convolve_gauss, mpl_hist, calc_background, discrete_interpolated_median, shade_plot, \
    rolling_median, InteractivePlot
from JSB_tools.spe_reader import SPEFile, EfficiencyCalMixin, EnergyCalMixin, _rebin

# HERE = pytz.timezone('US/Mountain')
OLE_TIME_ZERO = datetime.datetime(1899, 12, 30, 0, 0, 0)
cwd = Path(__file__).parent


class OriginalDataMixin:
    def __init__(self):
        self.__erg_calibration = tuple(self.erg_calibration)

        self.__adc_values = np.array(self.adc_values)
        self.__adc_values.flags.writeable = False

        self.__n_adc_channels = self.n_adc_channels

    def __recover__(self):
        self.erg_calibration = self.__erg_calibration
        self.adc_values = self.__adc_values
        self.n_adc_channels = self.__n_adc_channels


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

    spe_lines = ["$SPEC_ID:", sample_des, '$SPEC_REM:', f'DET# {_l.detector_id}',
                 f'DETDESC# {_l.device_address}', 'AP# Maestro Version', '$DATE_MEA:', datetime_str, '$MEAS_TIM:',
                 f'{_l.livetimes[-1]:.4f} {_l.realtimes[-1]:.4f}', '$DATA:', f'0 {_l.n_adc_channels - 1}']
    for counts in _l.get_erg_spectrum():
        spe_lines.append(f'       {counts}')
    fmt = lambda x: f'{x:.6E}'
    erg_cal_line = ' '.join(map(lambda x: f'{x:.6f}', _l.erg_calibration[:2]))
    mca_cal_line = ' '.join(map(fmt, _l.erg_calibration))
    mca_cal_line += f' {_l.erg_units}'
    shape_cal_line = ' '.join(map(fmt, _l.shape_cal))
    spe_lines.extend(map(str, ['$ROI:', '0', '$PRESETS:', 'None', '0', '0','$ENER_FIT:',
                               erg_cal_line, '$MCA_CAL:', len(_l.erg_calibration), mca_cal_line,
                         '$SHAPE_CAL:', len(_l.shape_cal), shape_cal_line]))
    return spe_lines


class MaestroListFile(EfficiencyCalMixin, EnergyCalMixin, OriginalDataMixin):
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
        self.path = path
        self._spe: SPEFile = None

        super(MaestroListFile, self).__init__()

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
            self.valid_erg_cal = self.read('1s', f)
            self.erg_units = self.read("4s", f)
            _erg_calibration = [self.read('f', f) for i in range(3)]
            super(EfficiencyCalMixin, self).__init__(_erg_calibration, load_erg_cal=False)

            self.valid_shape_cal = self.read('1s', f)
            self.shape_cal = [self.read('f', f) for i in range(3)]
            self.n_adc_channels = self.read('i', f)
            self.detector_id = self.read('i', f)
            self.total_realtime = self.read('f', f)
            self.total_livetime = self.read('f', f)
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
            self.count_rate_meter = []

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

        self.count_rate_meter = np.array(self.count_rate_meter) / 10E-3

        if debug:
            print("Start time: ", self.start_time)
            print("Live time: ", self.total_livetime)
            print("Real time: ", self.total_realtime)
            print("Total % live: ", self.total_livetime / self.total_realtime)
            print("N events/s = ", len(self.times) / self.total_realtime)
            print(f"Total counts: {self.n_counts}")

        self.adc_values = np.array(self.adc_values)
        t_log = time.time()
        if _print_progress:
            print('Converting pulse height data to energy...', end='')
        self._energies = self.channel_to_erg(self.adc_values)
        if _print_progress:
            print(f'Done. ({time.time() - t_log:.1f} seconds)')
        if _print_progress:
            t_log = time.time()
            print("Done.\nCalculating livetime fractions...", end='')

        self._fraction_live = None

        if _print_progress:
            print(f'Done. ({time.time() - t_log:.1f} seconds)')

        if _print_progress:
            t_log = time.time()
            print("Converting data to numpy arrays...", end='')
        self.sample_ready_state = np.array(self.sample_ready_state)
        self.gate_state = np.array(self.gate_state)
        self.times = np.array(self.times)
        self._energies = np.array(self._energies)
        self.realtimes = np.array(self.realtimes)
        self.livetimes = np.array(self.livetimes)

        # self.times.flags.writeable = False
        self.realtimes.flags.writeable = False
        self.livetimes.flags.writeable = False

        if _print_progress:
            print(f'Done. ({time.time() - t_log:.1f} seconds)')

        self._energy_binned_times = None
        self._energy_spec = None

        self.unpickle_eff()

        if debug:
            plt.plot(self.realtimes, self.livetimes)
            plt.xlabel("Real-time [s]")
            plt.ylabel("Live-time [s]")
            plt.figure()

            plt.plot(self.realtimes, self.fraction_live)
            plt.title("")
            plt.xlabel("Real-time [s]")
            plt.ylabel("% live-time")

        self.allow_pickle = True
        super(EnergyCalMixin, self).__init__()

    def set_useful_energy_range(self, erg_min=None, erg_max=None):
        if erg_min is None:
            cut1 = True
        else:
            cut1 = self.energies > erg_min
        if erg_max is None:
            cut2 = True
        else:
            cut2 = self.energies < erg_max
        s = np.where(cut1 & cut2)
        self.times = self.times[s]
        self._energies = self.energies[s]
        self.adc_values = self.adc_values[s]

    def __reset_properties__(self,):
        """
        Reset cached data related to energy, thus forcing code to recalculate from raw channel data.
        """
        self._spe = None
        self._energies = None
        self._energy_binned_times = None
        self._energy_spec = None
        if not hasattr(self, "_no_channels"):  # see doc
            for name in ['erg_bins', 'erg_centers']:
                try:
                    delattr(self, name)
                except AttributeError:
                    continue

    @property
    def fraction_live(self):
        if self._fraction_live is None:
            dead_time_corr_window = 10
            self._fraction_live = self._calc_fraction_live(self.livetimes, self.realtimes, dead_time_corr_window)
        return self._fraction_live

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
        # self.times.flags.writeable = True
        self.times = self.times + t
        # self.times.flags.writeable = False

        self._energy_binned_times = None
        self._energy_spec = None

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

    @property
    def SPE(self) -> SPEFile:
        """
        Get SPE object from Lis data.
        Returns:

        """

        if self._spe is None:
            return self.build_spe()
        return self._spe

    @property
    def energies(self):
        if self._energies is None:
            self._energies = self.channel_to_erg(self.adc_values)
            self._energies.flags.writeable = False
        return self._energies

    @property
    def erg_calibration(self):
        return self._erg_calibration

    @erg_calibration.setter
    def erg_calibration(self, coeffs):
        self._erg_calibration = np.array(coeffs)
        if self._energies is not None:
            old_energies = self.energies
        else:
            old_energies = None
        self._energies = None
        if self._spe is not None:
            self.SPE.erg_calibration = coeffs
        self.recalc_effs(old_energies)

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
            self.count_rate_meter.append(self.__10ms_counts__)
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
        s += f"valid_erg_calibration?: {bool(self.valid_erg_cal)}\n"
        s += f"Energy units: {self.erg_units}\n"
        s += f"Energy calibration: {self.erg_calibration}\n"
        s += f"valid_shape_cal: {bool(self.valid_shape_cal)}\n"
        s += f"Shape cal: {self.shape_cal}\n"
        s += f"# of adc channels: {self.n_adc_channels}\n"
        s += f"detector ID: {self.detector_id}\n"
        s += f"Maestro real time: {self.total_realtime}\n"
        s += f"Maestro live time: {self.total_livetime}\n"
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

    def get_time_dependence(self, energy,
                            bins: Union[str, int, np.ndarray, list] = 'auto',
                            signal_window_kev: float = 3,
                            bg_window_kev=None,
                            bg_offsets: Union[None, Tuple, float] = None,
                            make_rate=False,
                            eff_corr=False,
                            scale: Union[float, Callable, np.ndarray] = 1.,
                            nominal_values=True,
                            convolve: Union[float, int] = None,
                            offset_sample_ready=False,
                            debug_plot: Union[bool, str] = False):
        """
        Get the time dependence around erg +/- signal_window_kev/2. Baseline is estimated and subtracted.
        Estimation of baseline is done by taking the median rate ( with an energy of `energy` +/- `bg_window_kev`/2,
        excluding signal window) of events in each time bin. The median is then normalized to the width of the signal
        window (in KeV), and finally subtracted from the time dependence of events in the signal window.
        Args:
            energy:
            bins: Str/int for np.histogram or list of bin edges.
            signal_window_kev: Width around `energy` that will be considered the signal.
            bg_window_kev: Size of the window used for baseline estimation.
            bg_offsets: Offset the baseline window from the center (default will avoid signal window by distance of
                half `signal_window_kev`)
            make_rate: Makes units in Hz instead of counts.
            eff_corr: If True, account for efficiency
            scale: User supplied float or array of floats (len(bins) - 1) used to scale rates.
                If a Callable, the bins wil be passed to the supplied function which must return array of normalization
                values of same length as bins.
            nominal_values: If False, return unp.uarray (i.e. include Poissonian errors).
            convolve: If not None, perform gaussian convolution with sigma according to this value
                (sigma units are array indicies).
            offset_sample_ready: Subtract the median of the times for which SAMPLE READY port is ON.
            debug_plot: If False, do nothing.
                        If True, plot signal and background (energy vs counts) for every time bin.
                        If "simple", plot one plot for all bins.

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
        # bg_window_right_is[-1] += 1  # + 1 for python slicing rules
        # bg_window_left_is[-1] += 1  # + 1 for python slicing rules

        sig_window_is = self.__erg_index__(sig_window_bounds)
        # sig_window_is[-1] += 1  # + 1 for python slicing rules
        n_sig_erg_bins = sig_window_is[-1] - sig_window_is[0]

        if eff_corr:
            bg_weights = np.concatenate([1.0/unp.nominal_values(self.effs[slice(*bg_window_left_is)]),
                                         1.0/unp.nominal_values(self.effs[slice(*bg_window_right_is)])])

            sig_weight = np.average((1.0/self.effs[slice(*sig_window_is)]),
                                     weights=np.fromiter(map(len, self.__energy_binned_times__[slice(*sig_window_is)]),
                                                         dtype=int))
        else:
            bg_weights = 1
            sig_weight = ufloat(1, 0)

        bg_times_list = self.__energy_binned_times__[slice(*bg_window_left_is)]
        bg_times_list += self.__energy_binned_times__[slice(*bg_window_right_is)]

        sig_times_list = self.__energy_binned_times__[slice(*sig_window_is)]
        sig_times = np.concatenate(sig_times_list)

        # sig_times, n_sig_erg_bins = self.energy_slice(*sig_window_bounds, return_num_bins=True)

        if isinstance(bins, str):
            _, time_bins = np.histogram(sig_times, bins=bins)
        elif isinstance(bins, int):
            time_bins = np.linspace(sig_times[0], sig_times[-1], bins)
        else:
            assert hasattr(bins, '__iter__')
            time_bins = bins

        assert len(time_bins) != 1, "Only one bin specified. This is not allowed. "

        def _median(_x):
            # _x = convolve_gauss(_x, min([len(_x)//2, 5]))
            out = discrete_interpolated_median(_x*bg_weights)
            if nominal_values:
                return out
            else:
                return ufloat(out, out*1/np.sqrt(sum(_x)))

        bg = np.array([_median(x) for x in
                       np.array([np.histogram(times, bins=time_bins)[0] for times in bg_times_list]).transpose()])
        bg *= n_sig_erg_bins  # from per_bin to per width of signal window

        sig = np.histogram(sig_times, bins=time_bins)[0]
        sig = sig.astype(float)

        if not nominal_values:
            sig = unp.uarray(sig, np.sqrt(sig))

        sig *= sig_weight.n if nominal_values else sig_weight

        if debug_plot is not False:
            if isinstance(debug_plot, str) and debug_plot.lower() == 'simple':
                ax, y = self.plot_erg_spectrum(bg_window_left_bounds[0] - signal_window_kev,
                                               bg_window_right_bounds[-1] + signal_window_kev,
                                               time_min=time_bins[0],
                                               time_max=time_bins[-1],
                                               return_bin_values=True,
                                               eff_corr=eff_corr,
                                               )
                shade_plot(ax, bg_window_left_bounds, color='red')
                shade_plot(ax, bg_window_right_bounds, color='red', label='Bg. window')
                shade_plot(ax, sig_window_bounds, color='blue', label='Sig. window')
                ax.legend()
            else:
                axs = None
                for bin_index, (b1, b2) in enumerate(zip(time_bins[:-1], time_bins[1:])):
                    plot_legend = False
                    if bin_index % 4 == 0:
                        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
                        fig.suptitle(self.path.name)
                        axs = axs.flatten()
                        plt.subplots_adjust(hspace=0.4, wspace=0.1, left=0.04, right=0.97)
                        plot_legend = True

                    lines = []
                    legends = []
                    ax = axs[bin_index%4]

                    ax, y = self.plot_erg_spectrum(bg_window_left_bounds[0] - signal_window_kev,
                                                   bg_window_right_bounds[-1] + signal_window_kev,
                                                   time_min=b1,
                                                   time_max=b2,
                                                   return_bin_values=True,
                                                   ax=ax,
                                                   eff_corr=eff_corr)

                    l = shade_plot(ax, sig_window_bounds)
                    lines.append(l)
                    legends.append('Sig. window')

                    l = shade_plot(ax, bg_window_right_bounds, color='red')
                    _ = shade_plot(ax, bg_window_left_bounds, color='red')
                    lines.append(l)
                    legends.append('Bg. window')

                    bg_est = bg[bin_index] / n_sig_erg_bins
                    sig_est = sig[bin_index] / n_sig_erg_bins
                    if isinstance(bg_est, UFloat):
                        bg_est = bg_est.n
                        sig_est = sig_est.n

                    l = ax.plot(bg_window_right_bounds, [bg_est] * 2, color='red', ls='--')[0]
                    _ = ax.plot(bg_window_left_bounds, [bg_est] * 2,  color='red', ls='--')
                    lines.append(l)
                    legends.append('Bg. est.')

                    l = ax.plot(sig_window_bounds, [sig_est] * 2, color='blue', ls='--')[0]
                    lines.append(l)
                    legends.append('sig. est.')

                    if plot_legend:
                        plt.figlegend(lines, legends, 'upper right')

                    ax.text(np.mean(bg_window_left_bounds), ax.get_ylim()[-1]*0.82, f'{bg_est*n_sig_erg_bins:.2e}',
                            horizontalalignment='center',
                            verticalalignment='center', color='red', size='small', rotation='vertical')
                    ax.text(np.mean(sig_window_bounds), ax.get_ylim()[-1] * 0.82, f'{sig_est*n_sig_erg_bins:.2e}',
                            horizontalalignment='center',
                            verticalalignment='center', color='blue', size='small', rotation='vertical')

        if make_rate:
            if not isinstance(time_bins, np.ndarray):
                time_bins = np.array(time_bins)
            b_widths = time_bins[1:] - time_bins[:-1]
            sig /= b_widths
            bg /= b_widths

        if convolve is not None:
            sig, bg = convolve_gauss(sig, convolve), convolve_gauss(bg, convolve)

        if offset_sample_ready:
            time_bins -= self.sample_ready_median

        if hasattr(scale, '__call__'):
            c = scale(time_bins)
        else:
            if hasattr(scale, '__len__'):
                assert len(scale) == len(time_bins) - 1, '`normalization` argument of incorrect length.'
            c = scale

        return (sig - bg)*c, bg*c, time_bins

    def plot_time_dependence(self, energy, bins: Union[str, int, np.ndarray] = 'auto', signal_window_kev: float = 3,
                             bg_window_kev=None, bg_offsets: Union[None, Tuple, float] = None, make_rate=False,
                             eff_corr=False,
                             plot_background=False, ax=None, offset_sample_ready=False,
                             convolve=None,
                             debug_plot=False, **mpl_kwargs):
        """
        Todo: finish this doc string
        Args:
            energy:
            bins:
            signal_window_kev:
            bg_window_kev:
            bg_offsets:
            make_rate:
            eff_corr:
            plot_background:
            ax:
            offset_sample_ready:
            convolve:
            debug_plot:
            mpl_kwargs:

        Returns:

        """
        sig, bg, bins = \
            self.get_time_dependence(energy=energy, bins=bins, signal_window_kev=signal_window_kev,
                                     bg_window_kev=bg_window_kev, bg_offsets=bg_offsets, make_rate=make_rate,
                                     eff_corr=eff_corr, nominal_values=False, offset_sample_ready=offset_sample_ready,
                                     convolve=convolve, debug_plot=debug_plot)

        if ax is None:
            fig, ax = plt.subplots()
            if self.path is not None:
                ax.set_title(self.path.name)
                label = mpl_kwargs.pop("label", self.path.name)
            else:
                label = ''
            elims = energy - signal_window_kev/2, energy + signal_window_kev/2
            fig.suptitle(f"{elims[0]:.1f} < erg < {elims[-1]:.1f}")

        _, c = mpl_hist(bins, sig, return_line_color=True, ax=ax,
                        label=label + "(signal)" if plot_background else "", **mpl_kwargs)

        if plot_background:
            mpl_kwargs['ls'] = '--'
            mpl_kwargs.pop('c', None)
            mpl_kwargs.pop('color', None)
            mpl_kwargs['c'] = c
            mpl_hist(bins, bg, return_line_color=True, ax=ax, label=label + "(baseline)", **mpl_kwargs)

        y_label = "Raw" if not eff_corr else ""
        y_label += " Counts"

        if make_rate:
            y_label += '/s'

        ax.set_ylabel(y_label)
        ax.set_xlabel("time [s]")

        return ax

    def est_half_life(self, energy):
        raise NotImplementedError()

    @property
    def file_name(self):
        if isinstance(self.path, Path):
            return self.path.name
        elif self.path is not None:
            return self.path
        else:
            return 'None'

    def plot_percent_live(self, ax=None, **ax_kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.set_title(self.path.name)
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
        if isinstance(channel, list):
            channel = np.array(channel)
        return np.sum([channel ** i * c for i, c in enumerate(self.erg_calibration)], axis=0)

    @cached_property
    def erg_centers(self):
        return np.array([(b0+b1)/2 for b0, b1 in zip(self.erg_bins[:-1], self.erg_bins[1:])])

    @cached_property
    def erg_bins(self):
        channel_bins = np.arange(self.n_adc_channels + 1) - 0.5   #
        return self.channel_to_erg(channel_bins)

    @cached_property  # Also used as a standard attribute.
    def n_adc_channels(self):
        return max(self.adc_values)

    def plot_count_rate(self, ax=None, smooth=None, **ax_kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if 'ds' not in ax_kwargs:
            ax_kwargs['ds'] = 'steps-post'
        if smooth is not None:
            assert smooth > 0 and isinstance(smooth, int)
            smooth = int(smooth)
            y = convolve_gauss(self.count_rate_meter, smooth)
        else:
            y = self.count_rate_meter
        ax.plot(self.realtimes, y, **ax_kwargs)
        ax.set_title(self.path.name)
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
        # msg = True
        if self._energy_binned_times is None:
            self._energy_binned_times = [[] for _ in range(len(self.erg_bins) - 1)]
            # erg_indicies = self.__erg_index__(self.energies)
            for i, t in zip(self.__erg_index__(self.energies), self.times):
                if i >= self.n_adc_channels:
                    # if msg:
                    #     warnings.warn("TODO: I am a bug caused by rebinning. Find and kill me")
                    #     msg = False
                    continue
                self._energy_binned_times[i].append(t)
            self._energy_binned_times = [np.array(ts) for ts in self._energy_binned_times]
            self._energy_spec = None  # Force _energy_spec to update on next call
            return self._energy_binned_times
        else:
            return self._energy_binned_times

    @property
    def energy_spec(self):
        """
        Just the number of counts in each energy bin. Redundant. Just a special case of
            self.get_erg_spectrum (but faster).
        Returns:

        """
        if self._energy_spec is None:
            self._energy_spec = np.array(list(map(len, self.__energy_binned_times__)))
        return self._energy_spec

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
                         time_max: float = None, rebin=1, eff_corr=False, make_density=False, nominal_values=False,
                         return_bin_edges=False,
                         remove_baseline=False, baseline_method='ROOT', baseline_kwargs=None):
        """
        Get energy spectrum according to provided condition.
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            rebin: Combine bins into rebin groups
            eff_corr: If True, correct for efficiency to get absolute counts.
            make_density: If True, divide by bin widths.
            nominal_values: If False, return unp.uarray
            return_bin_edges: Return the energy bins, e.g. for use in a histogram.
            remove_baseline: If True, remove baseline according to the following arguments.
            baseline_method: If "ROOT" then use JSB_tools.calc_background
                             If "median" then use JSB_tools.rolling_median
            baseline_kwargs: Kwargs ot pass to either JSB_tools.calc_background or JSB_tools.rolling_median

        Returns:

        """
        if erg_min is None:
            erg_min = self.erg_bins[0]
        if erg_max is None:
            erg_max = self.erg_bins[-1]

        b = (time_min is not None, time_max is not None)

        if b == (0, 0):
            get_n_events = len
        elif b == (1, 0):
            def get_n_events(x):
                return len(x) - np.searchsorted(x, time_min, side='left')
        elif b == (0, 1):
            def get_n_events(x):
                return np.searchsorted(x, time_max, side='right')
        else:
            def get_n_events(x):
                return np.searchsorted(x, time_max, side='right') - np.searchsorted(x, time_min,
                                                                                    side='left')

        erg_index0 = self.__erg_index__(erg_min)
        erg_index1 = self.__erg_index__(erg_max)

        time_arrays = self.__energy_binned_times__[erg_index0: erg_index1]
        func = get_n_events
        out = np.fromiter(map(func, time_arrays), dtype=int)

        bins = self.erg_bins_cut(erg_min, erg_max)

        if not nominal_values:
            out = unp.uarray(out, np.sqrt(out))

        if make_density:
            out = out/(bins[1:] - bins[:-1])

        if remove_baseline:
            if baseline_kwargs is None:
                baseline_kwargs = {}
            baseline_method = baseline_method.lower()
            if baseline_method == 'root':
                out = out - calc_background(out, **baseline_kwargs)
            elif baseline_method == 'median':
                if 'window' not in baseline_kwargs:
                    baseline_kwargs['window_width'] = 75  # size of rolling window in keV
                out = out - rolling_median(values=out, **baseline_kwargs)
            else:
                raise TypeError(f"Invalid `baseline_method`, '{baseline_method}'")

        if eff_corr:
            assert self.effs is not None, 'Cannot perform efficiency correction. No efficiency data. '
            effs = self.effs[erg_index0: erg_index1]
            if nominal_values:
                effs = unp.nominal_values(effs)
                if not out.dtype == 'float64':
                    out = out.astype('float64')
            out /= effs

        if rebin != 1:
            out, bins = _rebin(rebin, out, bins)

        if return_bin_edges:
            return out, bins
        else:
            return out

    @property
    def bin_centers(self):
        return 0.5*(self.erg_bins[1:] + self.erg_bins[:-1])

    @property
    def bin_widths(self):
        return self.erg_bins[1:] - self.erg_bins[:-1]

    def rebin(self, new_bins):
        """

        """
        select = np.ones(len(self.energies), dtype=bool)
        if new_bins[0] < self.erg_bins[0]:
            select *= np.where(self.energies < new_bins[0], False, True)
        if new_bins[-1] >= self.erg_bins[-1]:
            select *= np.where(self.energies >= new_bins[-1], False, True)
        energies = self.energies[select]

        np.random.seed(1)
        rnds = np.random.random(len(energies)) - 0.5
        idxs = self.__erg_index__(energies)

        # self._energies = energies + rnds * self.bin_widths[idxs]
        self.adc_values = energies + rnds * self.bin_widths[idxs]
        self.erg_bins = new_bins
        self.erg_calibration = [0, 1]
        try:
            del self.erg_centers
        except AttributeError:
            pass
        #
        #
        # self.adc_values = energies
        # self.erg_calibration = [0, 1]
        # self.times = self.times[select]
        # self.energies
        #
        # self.__energy_binned_times__ = self._energies = None


    def merge_bins(self, n):
        ergs = np.mean(np.split(self.erg_centers[:n*(len(self.erg_centers) // n)], len(self.erg_centers)//n), axis=1)
        new_adc_values = self.adc_values//n
        new_erg_bins = self.erg_bins[::n]
        self.n_adc_channels = max(new_adc_values)
        selection = np.where(new_adc_values < self.n_adc_channels)

        self.adc_values = ergs[new_adc_values[selection].astype(int)]
        self.times = self.times[selection]
        self.erg_calibration = [0, 1]
        self._no_channels = True
        self.__reset_properties__()
        self.erg_bins = new_erg_bins
        self.erg_centers = 0.5*(self.erg_bins[1:] + self.erg_bins[:-1])

        self.n_adc_channels = len(self.erg_bins) - 1

        # for c in self._erg_calibration[2:]:
        #     assert c == 0, 'Todo'

        # self._erg_calibration[0] = (n/2. - 0.5)*self._erg_calibration[1] + self._erg_calibration[0]
        # self._erg_calibration[1] = n * self._erg_calibration[1]

    def plot_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = 0,
                          time_max: float = None, rebin=1, eff_corr=False, remove_baseline=False, make_density=False,
                          title=None, ax=None,
                          label=None,
                          return_bin_values=False):
        """
        Plot energy spectrum with time and energy cuts.
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            rebin: Combine n bins
            eff_corr: If True, correct for efficiency to get absolute counts.
            remove_baseline: If True, remove baseline.
            make_density:
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

        bin_values, bins = self.get_erg_spectrum(erg_min, erg_max, time_min, time_max, make_density=make_density,
                                                 return_bin_edges=True, rebin=rebin, eff_corr=eff_corr)
        if remove_baseline:
            bl = calc_background(bin_values)
            bin_values = -bl + bin_values

        mpl_hist(bins, bin_values, ax=ax, label=label)
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

    @staticmethod
    def multi_plotly(list_objs: List[MaestroListFile], scales=None, leg_labels=None, erg_min=40, erg_max=None, eff_corr=True,
                     time_bins=None, time_bin_width=15,
                     time_step: int = 5, time_min=None, time_max=None, remove_baseline=False,
                     nominal_values=True):
        """
        Plots energy spectra at different time intervals according to an interactive slider.
        Args:
            list_objs: series of MaestroListFile objects
            scales: List of normalizations of each list_obj
            leg_labels: None,
            erg_min:
            erg_max:
            eff_corr:
            time_bins:
            time_bin_width:
            time_step:
            time_min:
            time_max:
            remove_baseline:
            nominal_values:

        Returns:

        """
        _plt: InteractivePlot = None
        if scales is None:
            scales = np.ones(len(list_objs))

        if leg_labels is None:
            leg_labels = [None]*len(list_objs)

        assert len(leg_labels) == len(list_objs)
        attribs = None

        for l, label, s in zip(list_objs, leg_labels, scales):
            if attribs is None:
                attribs = l.plotly(erg_min=erg_min, erg_max=erg_max, eff_corr=eff_corr, time_bins=time_bins,
                                   time_bin_width=time_bin_width, time_step=time_step, time_min=time_min,
                                   time_max=time_max, remove_baseline=remove_baseline, nominal_values=nominal_values,
                                   leg_label=label, dont_plot=True
                                   )
            else:
                l.plotly(**attribs, eff_corr=eff_corr, time_bin_width=time_bin_width, leg_label=label, scale=s,
                         dont_plot=True)

        attribs['interactive_plot'].plot()

    def plotly(self, erg_min=40, erg_max=None, eff_corr=True, time_bins=None, time_bin_width=15,
               time_step: int = 5, time_min=None, time_max=None, remove_baseline=False,
               interactive_plot: InteractivePlot = None, nominal_values=True, leg_label=None, scale=1,
               dont_plot=False):
        """

        Args:
            erg_min:
            erg_max:
            eff_corr:
            time_bins:
            time_bin_width:
            time_step:
            time_min:
            time_max:
            remove_baseline:
            interactive_plot:
            nominal_values:
            leg_label:
            scale: Multiply y axis by this number.
            dont_plot:

        Returns:
            dictionary as follows
                {"interactive_plot": interactive_plot: JSB_tools.InteractivePlot,
                #  The rest of the key/values are arguments provided to plotly (they may be different if None wsa used).
                "remove_baseline": `remove_baseline`,
                "time_bins": time_bins,
                "erg_max": erg_max, "erg_min": erg_min,
                "nominal_values": nominal_values}

        """
        if time_bins is not None:
            time_bins = np.array(time_bins, copy=False)
            assert time_bins.ndim == 2 and time_bins.shape[1] == 2
            # assert None is time_bin_width is time_step is time_min is time_max,
        else:
            if time_max is None:
                time_max = self.times[-1]
            if time_min is None:
                time_min = -time_bin_width/2

            assert None not in [time_min, time_bin_width, time_step]
            time_max = time_min + time_step * ((time_max - time_min) // time_step)
            time_centers = np.arange(time_min, time_max + time_step, time_step)
            lbins = time_centers - time_bin_width / 2
            rbins = time_centers + time_bin_width / 2
            time_bins = list(zip(np.where(lbins >= time_min, lbins, time_min), np.where(rbins <= time_max, rbins, time_max)))

        time_bins = np.array(time_bins, copy=False)

        ys = []
        labels4frames = []
        assert len(time_bins) > 0

        for (b0, b1) in time_bins:
            _y, bin_edges = self.get_erg_spectrum(erg_min, erg_max, b0, b1, eff_corr=eff_corr,
                                                  nominal_values=nominal_values,
                                                  return_bin_edges=True)
            _y /= (b1 - b0)
            _y *= scale

            if remove_baseline:
                _y -= rolling_median(45, _y)
            ys.append(_y)
            labels4frames.append(f"{b0} <= t < {b1} ({0.5*(b0 + b1)})")

        if interactive_plot is None:
            interactive_plot = InteractivePlot(labels4frames, "time ")

        x = (bin_edges[1:] + bin_edges[:-1]) / 2

        if leg_label is None:
            leg_label = self.file_name

        color = interactive_plot.add_ys(x, ys, leg_label=leg_label, line_type='hist', return_color=True)

        tot_time = (time_bins[-1][-1] - time_bins[0][0])

        if nominal_values:
            tot_y = np.mean(ys, axis=0)
            error_y = None
        else:
            tot = np.mean(ys, axis=0)
            tot_y = unp.nominal_values(tot)
            error_y = unp.std_devs(tot)
        # tot_y = self.get_erg_spectrum(erg_min=erg_min, erg_max=erg_max, time_min=time_min, time_max=time_max,
        #                               remove_baseline=remove_baseline, nomin)

        interactive_plot.add_persistent(x, tot_y, yerr=error_y, leg_label='All time', opacity=0.5, line_type='hist')

        interactive_plot.fig.update_layout(
            scene=dict(
                yaxis=dict(range=[0, interactive_plot.max_y])))

        if not dont_plot:
            interactive_plot.plot()

        return {"interactive_plot": interactive_plot, "remove_baseline": remove_baseline, "time_bins": time_bins,
                "erg_max": erg_max, "erg_min": erg_min,
                "nominal_values": nominal_values}

    # def plotly(self, erg_min=None, erg_max=None, erg_bins='auto', eff_corr=True, time_bin_width=15,
    #            time_step: int = 5, time_min=0, time_max=None, percent_of_max=False, remove_baseline=False, title=None):
    #     """
    #     Args:
    #         erg_min:
    #         erg_max:
    #         erg_bins:
    #         eff_corr:
    #         time_bin_width: Width of time range for each slider step.
    #         time_step: delta t between each slider step
    #         time_min: Min time
    #         time_max: Max time for plot.
    #         percent_of_max: It True, each bin is fraction of max.
    #         remove_baseline:
    #
    #     Returns:
    #
    #     """
    #     if erg_min is None:
    #         erg_min = self.erg_bins[0]
    #     if erg_max is None:
    #         erg_max = self.erg_bins[-1]
    #     if title is None:
    #         title = self.file_name if self.file_name is not None else "untitled"
    #
    #     if time_max is None:
    #         time_max = self.times[-1]
    #
    #     # time_centers = np.linspace(time_min, time_max, int((time_max-time_min) // time_step + 1))
    #     time_max = time_min + time_step*((time_max - time_min)//time_step)
    #     time_centers = np.arange(time_min, time_max, time_step)
    #     time_groups = [(max([time_min, b - time_bin_width / 2]), min([time_max, b + time_bin_width / 2])) for b in time_centers]
    #     time_bin_widths = [b1-b0 for b0, b1 in time_groups]
    #
    #     energy_bins = self.erg_bins_cut(erg_min, erg_max)
    #     energy_bin_centers = (energy_bins[1:] + energy_bins[:-1])/2
    #     labels4bins = []
    #
    #     ys = []
    #
    #     for b_width, (b0, b1) in zip(time_bin_widths, time_groups):
    #         _y = self.get_erg_spectrum(erg_min, erg_max, b0, b1, eff_corr=eff_corr, nominal_values=True) / b_width
    #         if remove_baseline:
    #             _y -= rolling_median(45, _y)
    #         ys.append(_y)
    #         labels4bins.append((b0, b1))
    #
    #     ys = np.array(ys)
    #
    #     fig = go.Figure()
    #
    #     y_tot = self.get_erg_spectrum(erg_min, erg_max, time_max=time_max, nominal_values=True,
    #                                   remove_baseline=remove_baseline, eff_corr=eff_corr)/(time_max-time_min)
    #
    #     fig.add_trace(
    #         go.Bar(
    #             visible=True,
    #             name=f"All time",
    #             x=energy_bin_centers,
    #             y=y_tot,
    #             marker_color='red',
    #         )
    #     )
    #
    #     steps = [dict(
    #             method="update",
    #             args=[{"visible": [True] + [False] * 2*len(time_centers)},
    #                   {"title": f"{title} All time"}],  # layout attribute
    #         )]
    #
    #     _max_y = 0
    #     if percent_of_max:
    #         ys /= np.array([np.max(convolve_gauss(_y, 3)) for _y in ys])[:, np.newaxis]
    #
    #     for index, (y, (t0, t1)) in enumerate(zip(ys, time_groups)):
    #         b_width = t1 - t0
    #         b_center = (t1+t0)/2
    #         if max(y)>_max_y:
    #             _max_y = max(y)
    #
    #         label = f"t ~= {b_center:.1f} [s] ({t0:.1f} < t < {t1:.1f})"
    #         assert len(energy_bin_centers) == len(y), [len(energy_bin_centers), len(y)]
    #         fig.add_trace(
    #             go.Scatter(
    #                 visible=False,
    #                 x=energy_bin_centers,
    #                 y=y,
    #                 marker_color='blue',
    #                 line={'shape': 'hvh'}
    #             ),
    #         )
    #         fig.add_trace(
    #             go.Scatter(
    #                 visible=False,
    #                 x=energy_bin_centers,
    #                 y=convolve_gauss(y, 3),
    #                 ))
    #         step = dict(
    #             method="update",
    #             args=[{"visible": [False] * (2*len(time_centers) + 1)},
    #                   {"title": f"{title} {label}"}],  # layout attribute
    #         )
    #         step["args"][0]["visible"][2*index+1] = True  # Toggle trace to "visible"
    #         step["args"][0]["visible"][2*index+2] = True  # Toggle trace to "visible"
    #         steps.append(step)
    #     fig.update_yaxes(range=[0, _max_y*1.1])
    #     sliders = [dict(
    #         active=0,
    #         currentvalue={"prefix": "time "},
    #         pad={"t": 50},
    #         steps=steps
    #     )]
    #
    #     fig.update_layout(
    #         sliders=sliders, bargap=0, bargroupgap=0.0
    #     )
    #     fig.show()
    #     return fig

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

    def build_spe(self, min_time=None, max_time=None):
        """
        Construct and return the SPEFile instance from list data.
        Args:
            min_time: Min time cut off
            max_time: Max time cut off

        Returns:

        """
        if min_time is None and max_time is None:
            counts = self.energy_spec
            set_spe = True  # if True, save the result so that it isn't built again this session
            total_livetime = self.total_livetime
            total_realtime = self.total_realtime
        else:
            # select counts.
            if min_time is None:
                min_time = 0
            if max_time is None:
                max_time = self.times[-1]
            counts = self.get_erg_spectrum(time_min=min_time, time_max=max_time)
            set_spe = False

            # adjust total real and live time
            if self.realtimes is not self.livetimes is not None:
                rt_i_max = len(self.realtimes) - 1
                rt_i_min = 0
                if max_time is not None:
                    rt_i_max = np.searchsorted(self.realtimes, max_time)
                    rt_i_max = min([len(self.realtimes)-1, rt_i_max])

                if min_time is not None:
                    rt_i_min = np.searchsorted(self.realtimes, min_time)
                assert rt_i_max != rt_i_min

                total_livetime = self.livetimes[rt_i_max] - self.livetimes[rt_i_min]
                total_realtime = self.realtimes[rt_i_max] - self.realtimes[rt_i_min]
            else:
                total_realtime = max_time - min_time
                total_livetime = total_realtime

        path = self.path.with_suffix('.Spe') if self.path is not None else None
        out = SPEFile.build(path=path, counts=counts,
                            erg_calibration=self._erg_calibration, livetime=total_livetime, realtime=total_realtime,
                            channels=np.arange(self.n_adc_channels), erg_units=self.erg_units, shape_cal=self.shape_cal,
                            description=self.description, system_start_time=self.start_time, eff_model=self.eff_model,
                            effs=self.effs, eff_scale=self.eff_scale, load_erg_cal=self._load_erg_cal)

        if set_spe:
            self._spe = out
        return out

    def __set_optional_attribs_defaults__(self):
        """
        Sets unimportant attributes and place holders to their defaults. The vast majority of class features will work
         without theses attributes. Does not modify attributes if they exist.

        Returns: None
        """
        attribs = ['max_words', 'count_rate_meter', 'device_address', 'list_style', 'description', 'valid_erg_cal',
                   'serial_number', 'MCB_type_string', 'erg_units', 'shape_cal', 'detector_id', 'count_rate_meter',
                   '_energy_binned_times', '_energy_spec', '_spe', '_fraction_live', '_energies']
        for name in attribs:
            if not hasattr(self, name):
                setattr(self, name, None)

        self.allow_pickle = True

    def pickle(self, f_path=None, write_spe=True):
        # Change attribs to np.array where appropriate
        if not self.allow_pickle:
            raise ValueError("Pickling not allowed!")

        for name in ['_erg_calibration', 'adc_values', 'sample_ready_state', 'gate_state', 'realtimes', 'livetimes',
                     'times']:
            a = getattr(self, name)
            if not isinstance(a, np.ndarray):
                setattr(self, name, np.array(a))

        if self.adc_values.dtype != float:
            self.adc_values = self.adc_values.astype(float, copy=False)

        d = {'times': self.times,
             'realtimes': self.realtimes,
             'livetimes': self.livetimes,
             'n_adc_channels': int(self.n_adc_channels),
             '_erg_calibration': list(map(float, self._erg_calibration)),
             'gate_state': self.gate_state,
             'sample_ready_state': self.sample_ready_state,
             'adc_values': self.adc_values,
             'path': str(self.path),
             'start_time': datetime.datetime.strftime(self.start_time, MaestroListFile.datetime_format),
             'total_realtime': float(self.total_realtime) if self.total_realtime is not None else None,
             'total_livetime': float(self.total_livetime) if self.total_livetime is not None else None,
             'count_rate_meter': self.count_rate_meter,
             '_fraction_live': self._fraction_live,
             }

        if hasattr(self, '_no_channels'):
            d['_no_channels'] = True
            d['erg_bins'] = np.array(list(self.erg_bins))

        d_np_types = {'gate_state': 'int', 'count_rate_meter': 'int', 'sample_ready_state': 'int'}
        if f_path is None:
            if self.path.name == '':
                raise ValueError("No 'self.path'. Either set path or include `f_path` argument.")
            f_path = self.path.parent / self.file_name
        f_path = Path(f_path).with_suffix('.marshal')

        with open(f_path, 'wb') as f:
            marshal.dump(d, f)
            marshal.dump(d_np_types, f)

        if write_spe:
            self.__set_optional_attribs_defaults__()
            self.SPE.pickle(f_path=f_path)
        else:
            self.pickle_eff(path=f_path)  # dont pickle eff twice

    @classmethod
    def from_pickle(cls, fpath, load_erg_cal=None) -> MaestroListFile:
        # todo: Re work this using a helper class! (same for Spefile)
        fpath = Path(fpath).with_suffix('.marshal')

        if not fpath.exists():
            raise FileNotFoundError(f'MaestroListFile marshal path does not exist, "{fpath}"')
        with open(fpath, 'rb') as f:
            d = marshal.load(f)
            d_np_types = marshal.load(f)

        self = MaestroListFile.__new__(MaestroListFile)
        super(MaestroListFile, self).__init__()  # run EfficiencyMixin init
        self.__set_optional_attribs_defaults__()
        for k, v in d.items():
            if isinstance(v, bytes):
                if k in d_np_types:
                    t = eval(d_np_types[k])
                else:
                    t = float
                v = np.frombuffer(v, dtype=t)
            elif isinstance(v, float):
                v = float(v)
            elif isinstance(v, int):
                v = int(v)
            setattr(self, k, v)

        if hasattr(self, '_no_channels'):
            self._energies = self.adc_values

        self.path = Path(self.path)
        self.start_time = datetime.datetime.strptime(self.start_time, MaestroListFile.datetime_format)
        super(EfficiencyCalMixin, self).__init__(self.erg_calibration, load_erg_cal=load_erg_cal)
        # self.times.flags.writeable = False

        self.unpickle_eff()

        return self

    @classmethod
    def build(cls, adc_values, times, n_adc_channels, erg_calibration, gate_state=None,
              sample_ready_state=None, path=None, start_time=None, realtimes=None, livetimes=None,
              total_livetime=None, total_realtime=None, effs=None, eff_model=None,
              load_erg_cal=None) -> MaestroListFile:
        self = MaestroListFile.__new__(MaestroListFile)
        super(MaestroListFile, self).__init__()  # run EfficiencyMixin init

        self.__set_optional_attribs_defaults__()
        self.adc_values = np.array(adc_values, dtype=float)
        self.n_adc_channels = n_adc_channels
        self.times = np.array(times)
        self._erg_calibration = list(map(float, erg_calibration))  # no np.float64
        if path is None:
            self.path = None
        else:
            self.path = Path(path)
        if start_time is None:
            start_time = datetime.datetime.now()
        self.start_time = start_time

        self.realtimes = realtimes
        self.livetimes = livetimes

        self.gate_state = np.array(gate_state) if gate_state is not None else gate_state
        self.sample_ready_state = np.array(sample_ready_state) if sample_ready_state is not None else sample_ready_state

        if self.sample_ready_state is None:
            if self.realtimes is None:
                self.sample_ready_state = np.array([0])
            else:
                self.sample_ready_state = np.zeros_like(self.realtimes)

        if total_realtime is None:
            total_realtime = times[-1]
        self.total_realtime = float(total_realtime) if total_realtime is not None else None
        self.total_livetime = float(total_livetime) if total_livetime is not None else None
        self.effs = effs
        self.eff_model = eff_model
        super(EfficiencyCalMixin, self).__init__(self.erg_calibration, load_erg_cal=load_erg_cal)
        self.allow_pickle = True
        super(EnergyCalMixin, self).__init__()
        return self

    def __len__(self):
        return len(self.times)

    def __iadd__(self, other: MaestroListFile, truncate=True):
        """
        Merge the events of another List file into this one.
        Note: Have not verified if the calculations for deadtime correction are correct.
        Args:
            other: A other list file.
            truncate: If True, truncate all events in the list file with the largest recorded time that occur at a
                time after the final recorded time in the other list file. I.e., if one list file was acquiring for
                longer than the other, it is usually beneficial to truncate the longer one so that they are the
                 same length.

        Returns:

        """
        assert isinstance(other, MaestroListFile)

        if not all(other.erg_bins == self.erg_bins):
            other.rebin(self.erg_bins)

        if truncate:
            tr = [self, other]
            larger_index = None
            if self.times[-1] > other.times[-1]:
                larger_index = 0
            elif self.times[-1] < other.times[-1]:
                larger_index = 1
            if larger_index is not None:
                larger_list = tr.pop(larger_index)
                smaller = tr[0]
                max_time = smaller.times[-1]
                max_i = np.searchsorted(larger_list.times, max_time)
                setattr(larger_list, 'times', larger_list.times[:max_i])
                setattr(larger_list, '_energies', larger_list.energies[:max_i])

        idxs = np.searchsorted(self.times, other.times)
        self._energies = np.insert(self.energies, idxs, other.energies)
        self.times = np.insert(self.times, idxs, other.times)

        if not (self.count_rate_meter is other.count_rate_meter is None):
            pass  # this is hard.
            # self._fraction_live = other.fraction_live*self.fraction_live/\
            #                       (self.count_rate_meter*other.fraction_live +
            #                        other.count_rate_meter*self.fraction_live)
        self._energy_binned_times = None
        self._energy_spec = None
        self.path = Path()  # prevent from overwriting pickle with default path.
        self.allow_pickle = False
        return self


def get_merged_time_dependence(list_files: List[MaestroListFile], energy,
                               time_bins: Union[str, int, np.ndarray] = 'auto',
                               take_mean=False,
                               signal_window_kev: float = 3,
                               bg_window_kev=None,
                               bg_offsets: Union[None, Tuple, float] = None,
                               normalization_constants=None,
                               make_rate=False,
                               nominal_values=True,
                               offset_sample_ready=False, ):
    if normalization_constants is None:
        normalization_constants = np.ones(len(list_files), dtype=float)
    bins_Set_flag = False

    cs = normalization_constants
    bg = None
    sig = None
    tot_time = 0
    for l, c in zip(list_files, cs):
        _sig, _bg, _bins = l.get_time_dependence(energy=energy, bins=time_bins, signal_window_kev=signal_window_kev,
                                                 bg_window_kev=bg_window_kev, bg_offsets=bg_offsets, make_rate=False,
                                                 scale=c, nominal_values=nominal_values,
                                                 offset_sample_ready=offset_sample_ready)

        tot_time += l.total_livetime
        if not bins_Set_flag:
            time_bins = _bins  # set bins
            sig = _sig
            bg = _bg
            bins_Set_flag = True
        else:
            sig += _sig
            bg += _bg
    if take_mean:
        sig /= len(list_files)
        bg /= len(list_files)
    if make_rate:
        sig /= tot_time
        bg /= tot_time

    return sig, bg, time_bins


if __name__ == '__main__':
    bins = [0, 1, 2, 3, 4, 5,]
    other_bins = [0.5, 2, 3,4,5]

    # l = MaestroListFile.from_pickle('/Users/burggraf1/PycharmProjects/PHELIX/2021/data/shot40[2].marshal')
    # ax = l.plot_erg_spectrum(make_density=True)
    # l.rebin(np.linspace(l.erg_bins[0], l.erg_bins[-1], 3000))
    # l.plot_erg_spectrum(make_density=True, ax=ax)
    # plt.show()
    # l.rebin()
    # l.rebin_energy()

