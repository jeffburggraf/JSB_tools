"""
MaestroListFile class for reading and analysing Ortec Lis files.
Todo:
    Implement a method for efficiency calibration
"""
from __future__ import annotations
import warnings
from JSB_tools.spectra import ListSpectra, EfficiencyCalMixin
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
import winfiletime
from functools import cached_property
import time
from uncertainties.core import UFloat, ufloat
from uncertainties import unumpy as unp
from JSB_tools import ProgressReport, convolve_gauss, mpl_hist, calc_background, discrete_interpolated_median, shade_plot, \
    rolling_median, InteractivePlot, _float
from numpy.core._exceptions import UFuncTypeError
from JSB_tools.spe_reader import SPEFile

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


class MaestroListFile(ListSpectra):
    pickle_attribs = '_erg_calibration', 'device_address', 'MCB_type_string', 'serial_number', \
                     'detector_id', 'total_realtime', 'total_livetime', 'count_rate_meter', 'n_adc_channels',\
                     'description', 'realtimes', 'livetimes', 'sample_ready_state', 'gate_state'

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
        self._spe: SPEFile = None

        assert path.exists(), f'List file not found,"{path}"'

        with open(path, 'rb') as f:
            lst_header = self.read('i', f)

            self.adc_channels = []
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
            self._erg_calibration = [self.read('f', f) for i in range(3)]

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

            self.__times = []
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
        print(f"Done reading list data. Rate: {int(len(self.__times)/(time.time()-t0)):.2g} events read per second.")

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
            print("N events/s = ", len(self.__times) / self.total_realtime)
            print(f"Total counts: {self.n_counts}")

        self.adc_channels = np.array(self.adc_channels)
        t_log = time.time()
        if _print_progress:
            print('Converting pulse height data to energy...', end='')

        self._fraction_live = None

        self.sample_ready_state = np.array(self.sample_ready_state)
        self.gate_state = np.array(self.gate_state)
        # self.times = np.array(self.times)

        self.realtimes = np.array(self.realtimes)
        self.livetimes = np.array(self.livetimes)

        self.realtimes.flags.writeable = False
        self.livetimes.flags.writeable = False

        erg_bins = self.channel_to_erg(self.channels_list)

        fraction_live = self._calc_fraction_live(self.livetimes, self.realtimes, 10)

        super(MaestroListFile, self).__init__(adc_channels=self.adc_channels, times=self.__times,
                                              erg_bins=erg_bins, path=path,
                                              fraction_live=fraction_live,
                                              fraction_live_times=self.realtimes,
                                              start_time=self.start_time)

    def set_useful_energy_range(self, erg_min=None, erg_max=None):
        pass  # todo
        # if erg_min is None:
        #     cut1 = True
        # else:
        #     cut1 = self.energies > erg_min
        # if erg_max is None:
        #     cut2 = True
        # else:
        #     cut2 = self.energies < erg_max
        # s = np.where(cut1 & cut2)
        # self.times = self.times[s]
        # self._energies = self.energies[s]
        # self.adc_values = self.adc_values[s]

    @property
    def channels_list(self):
        return np.arange(self.n_adc_channels + 1) - 0.5

    # @property
    # def fraction_live(self):
    #     if self._fraction_live is None:
    #         dead_time_corr_window = 10
    #         self._fraction_live = self._calc_fraction_live(self.livetimes, self.realtimes, dead_time_corr_window)
    #     return self._fraction_live

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

        self.reset_cach()

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
    def erg_calibration(self):
        return self._erg_calibration

    @erg_calibration.setter
    def erg_calibration(self, coeffs):
        self._erg_calibration = np.array(coeffs)
        self.erg_bins = self.channel_to_erg(self.channels_list)
        self.reset_cach()

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
            self.adc_channels.append(adc_value)
            n_200ns_ticks = int(word[16:], 2)
            adc_time = self.n_rollovers * 10E-3 + n_200ns_ticks * 200E-9
            self.__times.append(adc_time)
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
            w_time = winfiletime.to_datetime(w_time)
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

    # def get_time_dependence(self, energy,
    #                         bins: Union[str, int, np.ndarray, list] = 'auto',
    #                         signal_window_kev: float = 3,
    #                         bg_window_kev=None,
    #                         bg_offsets: Union[None, Tuple, float] = None,
    #                         make_rate=False,
    #                         eff_corr=False,
    #                         scale: Union[float, Callable, np.ndarray] = 1.,
    #                         nominal_values=True,
    #                         convolve: Union[float, int] = None,
    #                         offset_sample_ready=False,
    #                         debug_plot: Union[bool, str] = False):
    #     """
    #     Get the time dependence around erg +/- signal_window_kev/2. Baseline is estimated and subtracted.
    #     Estimation of baseline is done by taking the median rate ( with an energy of `energy` +/- `bg_window_kev`/2,
    #     excluding signal window) of events in each time bin. The median is then normalized to the width of the signal
    #     window (in KeV), and finally subtracted from the time dependence of events in the signal window.
    #     Args:
    #         energy:
    #         bins: Str/int for np.histogram or list of bin edges.
    #         signal_window_kev: Width around `energy` that will be considered the signal.
    #         bg_window_kev: Size of the window used for baseline estimation.
    #         bg_offsets: Offset the baseline window from the center (default will avoid signal window by distance of
    #             half `signal_window_kev`)
    #         make_rate: Makes units in Hz instead of counts.
    #         eff_corr: If True, account for efficiency
    #         scale: User supplied float or array of floats (len(bins) - 1) used to scale rates.
    #             If a Callable, the bins wil be passed to the supplied function which must return array of normalization
    #             values of same length as bins.
    #         nominal_values: If False, return unp.uarray (i.e. include Poissonian errors).
    #         convolve: If not None, perform gaussian convolution with sigma according to this value
    #             (sigma units are array indicies).
    #         offset_sample_ready: Subtract the median of the times for which SAMPLE READY port is ON.
    #         debug_plot: If False, do nothing.
    #                     If True, plot signal and background (energy vs counts) for every time bin.
    #                     If "simple", plot one plot for all bins.
    #
    #     Returns: Tuple[signal window rate, baseline rate estimation, bins used]
    #     """
    #     # if offset_sample_ready:
    #     #     self.time_offset(-self.sample_ready_median)
    #     super()
    #     # todo super() ...

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

        if self.path is not None:
            ax.set_title(self.path.name)

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

    # @property
    # def energy_spec(self):
    #     """
    #     Just the number of counts in each energy bin. Redundant. Just a special case of
    #         self.get_erg_spectrum (but faster).
    #     Returns:
    #
    #     """
    #     if self._energy_spec is None:
    #         self._energy_spec = np.array(list(map(len, self.energy_binned_times)))
    #     return self._energy_spec

    def channel_to_erg(self, channel) -> np.ndarray:
        if isinstance(channel, list):
            channel = np.array(channel)
        return np.sum([channel ** i * c for i, c in enumerate(self.erg_calibration)], axis=0)

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
        # todo after canges
        """
        Construct and return the SPEFile instance from list data.
        Args:
            min_time: Min time cut off
            max_time: Max time cut off

        Returns:

        """
        if min_time is None and max_time is None:
            counts = self.get_erg_spectrum()
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

        path = None
        if self.path is not None and len(self.path.name):
            path = self.path.with_suffix('.Spe')

        out = SPEFile.build(path=path, counts=counts,
                            erg_calibration=self._erg_calibration, livetime=total_livetime, realtime=total_realtime,
                            channels=np.arange(self.n_adc_channels),
                            description=self.description, system_start_time=self.start_time, shape_cal=[0, 1, 0], erg_units='keV')

        if set_spe:
            self._spe = out
        return out

    def pickle(self, path=None, meta_data=None):
        assert not (self.path is path is None), "A `path` argument must be supplied as it wasn't specified at " \
                                                "initialization. "
        if path is None:
            path = self.path

        path = Path(path)

        if meta_data is None:
            meta_data = {}

        for name in self.pickle_attribs:
            assert name not in meta_data, f"`meta_data` key, '{name}', is already an attribute of present class. " \
                                          f"No overwriting allowed here."

            meta_data[name] = getattr(self, name)

        super(MaestroListFile, self).pickle(path=path, meta_data=meta_data)

    @classmethod
    def from_pickle(cls, path, load_erg_cal=None) -> MaestroListFile:
        path = Path(path)
        self = super(cls, cls).from_pickle(path)
        self._spe = None

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
    old = MaestroListFile('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/tuesday/shot1.Lis')
    old.pickle()
    ax = old.plot_erg_spectrum()

    s = MaestroListFile.from_pickle('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/tuesday/shot1.Lis')
    s.plot_erg_spectrum(ax=ax)
    plt.show()
    print()
