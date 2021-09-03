from __future__ import annotations

import marshal
import struct
from struct import unpack, calcsize
from pathlib import Path
import datetime
from JSB_tools.TH1 import TH1F
import pytz
from bitstring import BitStream
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from datetime import timezone
from typing import List, Union
import filetime
from functools import cached_property
import time
from JSB_tools import ProgressReport, convolve_gauss, mpl_hist
from JSB_tools.spe_reader import SPEFile

HERE = pytz.timezone('US/Mountain')
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
    """

    @property
    def __default_spe_path__(self):
        """
        The default path the MaestroList creates when converting List to Spe
        Returns:

        """
        return self._original_path.with_name(f'_{self._original_path.name}').with_suffix('.Spe')

    def list2spe(self, save_path=None):
        if save_path is None:
            save_path = self.__default_spe_path__
            # if not overwrite:
            #     if save_path.with_suffix('.Spe').exists():
            #         save_path = save_path.with_name(f'_{save_path.with_suffix("").name}')
        else:
            save_path = Path(save_path)
        save_path = save_path.with_suffix(".Spe")
        spe_text = '\n'.join(get_spe_lines(self))
        with open(save_path, 'w') as f:
            f.write(spe_text)
        return SPEFile(save_path)

    def read(self, byte_format, f, debug=False):
        s = calcsize(byte_format)
        _bytes = f.read(s)
        # _bytes = self.bytes[self.index: self.index + s]
        # self.index += s
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
        p = self.__default_spe_path__
        if p.exists():
            return SPEFile(p)
        else:
            return self.list2spe()

    def set_energy_cal(self, *coeffs):
        self.erg_calibration = np.array(coeffs)
        self.energies = self.channel_to_erg(self.adc_values)
        self.SPE.set_energy_cal(coeffs)

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

            if self.adc_zero_datetime is None:  # only set this for the first time we see a Win64 time.
                self.adc_zero_datetime = w_time

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
        self.file_name = path.name
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
            self.start_time = ole2datetime(self.read('d', f))
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
            self.adc_zero_datetime: Union[datetime.datetime, None] = None

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
            t0 = time.time()

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

        # kernel_size = 5*dead_time_corr_window
        # kernel = norm(loc=0, scale=dead_time_corr_window)\
        #     .pdf(np.linspace(-kernel_size//2, kernel_size//2, kernel_size))  # scale: 0.1 seconds
        # kernel /= np.sum(kernel)
        dead_time_corr_window = 10

        self.fraction_live = np.gradient(self.livetimes) / np.gradient(self.realtimes)
        self.fraction_live = convolve_gauss(self.fraction_live, dead_time_corr_window)
        # self.fraction_live = np.convolve(self.fraction_live, kernel, mode='same')

        # correct edge effects
        self.fraction_live[0:dead_time_corr_window // 2] = \
            np.median(self.fraction_live[dead_time_corr_window // 2:dead_time_corr_window])

        self.fraction_live[-dead_time_corr_window // 2:] = \
            np.median(self.fraction_live[-dead_time_corr_window: -dead_time_corr_window // 2])

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

            percent_live_hist = TH1F.from_raw_data(self.fraction_live, bins=100)
            ax = percent_live_hist.plot(show_stats=True, xlabel="% live-time", ylabel="Counts")
            ax.set_title("Frequencies of percent live-time")

    def plot_percent_live(self, ax=None, **ax_kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()
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

    def channel_to_erg(self, a):
        return np.sum([a ** i * c for i, c in enumerate(self.erg_calibration)], axis=0)

    @cached_property
    def erg_centers(self):
        return np.array([(b0+b1)/2 for b0, b1 in zip(self.erg_bins[:-1], self.erg_bins[1:])])

    @cached_property
    def erg_bins(self):
        channel_bins = np.arange(self.n_adc_channels)
        return self.channel_to_erg(channel_bins)

    def erg_bin_index(self, erg):
        return np.searchsorted(self.erg_bins, erg, side='right') - 1

    def get_spectrum_hist(self, t1=0, t2=None) -> TH1F:
        if t2 is None:
            index2 = len(self.times)
        else:
            index2 = np.searchsorted(self.times, t2)
        index1 = np.searchsorted(self.times, t1)
        hist = TH1F(bin_left_edges=self.erg_bins)
        for erg in self.energies[index1: index2]:
            hist.Fill(erg)
        return hist

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
        Get index of energy bin(s) from list of energies
        Args:
            energies:

        Returns:

        """
        return np.searchsorted(self.erg_bins, energies, side='right') - 1

    @property
    def __energy_binned_times__(self):
        """
        The times of all events are segregated into the available energy bins.
        Returns:

        """
        if self.__needs_updating__ or self._energy_binned_times is None:
            self._energy_binned_times = [[] for _ in range(len(self.erg_centers))]
            erg_indicies = self.__erg_index__(self.energies)
            for i, t in zip(erg_indicies, self.times):
                self._energy_binned_times[i].append(t)
            self._energy_binned_times = [np.array(ts) for ts in self._energy_binned_times]
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
        return self.erg_bins[np.where((self.erg_bins >= erg_min) & (self.erg_bins <= erg_max))]

    def get_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = 0,
                         time_max: float = None,):
        energies = self.energies
        if time_max is not None or time_min != 0:
            if time_max is None:
                time_max = self.times[-1]
            energies = self.energies[np.where((self.times >= time_min) & (self.times <= time_max))]
        if erg_min is None:
            erg_min = self.erg_bins[0]
        if erg_max is None:
            erg_max = self.erg_bins[-1]
        bin_values, bins = np.histogram(energies, bins=self.erg_bins_cut(erg_min, erg_max))
        return bin_values, bins

    def plot_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = 0,
                          time_max: float = None, ax=None):
        """
        Plot energy spectrum with time and energy cuts.
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            ax:

        Returns:

        """
        if ax is None:
            plt.figure()
            ax = plt.gca()

        bin_values, bins = self.get_erg_spectrum(erg_min, erg_max, time_min, time_max)
        mpl_hist(bins, bin_values, np.sqrt(bin_values), ax=ax)
        ax.set_title(f"{time_min} <= t <= {time_max}")
        ax.set_xlabel('Energy [KeV]')
        ax.set_ylabel('Counts')
        return ax

    def get_energies_in_range(self, erg_min, erg_max):
        """
        Integrate events over energy.
        Args:
            erg_min:
            erg_max:

        Returns:

        """
        return self.energies[np.where((self.energies >= erg_min) & (self.energies <= erg_max))]

    def get_times_in_range(self, erg_min, erg_max):
        """
        Return the times of all events with energy greater than `erg_min` and less than `erg_max`
        Args:
            erg_min:
            erg_max:

        Returns:

        """
        return np.concatenate(self.__energy_binned_times__[self.__erg_index__(erg_min): self.__erg_index__(erg_max)+1])

    def pickle(self, f_path=None):
        d = {'times': self.times, 'max_words': self.max_words, 'realtimes': self.realtimes,
             'livetimes': self.livetimes, 'n_adc_channels': self.n_adc_channels, 'fraction_live': self.fraction_live,
             'erg_calibration': self.erg_calibration, 'gate_state': self.gate_state,
             'sample_ready_state': self.sample_ready_state, 'adc_values': self.adc_values,
             'counts_per_sec': self.counts_per_sec, '__needs_updating__': True,
             '_original_path': str(self._original_path),
             'start_time': datetime.datetime.strftime(self.start_time, MaestroListFile.datetime_format),
             'device_address': self.device_address, 'list_style': self.list_style, 'description': self.description,
             'valid_erg_calibration': self.valid_erg_calibration, 'serial_number': self.serial_number,
             'MCB_type_string': self.MCB_type_string, 'erg_units': self.erg_units,
             'shape_calibration': self.shape_calibration, 'det_id_number': self.det_id_number,
             'total_real_time': self.total_real_time, 'total_live_time': self.total_live_time,
             'adc_zero_datetime': datetime.datetime.strftime(self.adc_zero_datetime, MaestroListFile.datetime_format)
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
        self.adc_zero_datetime = datetime.datetime.strptime(self.adc_zero_datetime, MaestroListFile.datetime_format)
        self.energies = self.channel_to_erg(self.adc_values)
        return self

    def __len__(self):
        return len(self.times)


if __name__ == '__main__':
    l = MaestroListFile('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot132.Lis')
    l.set_energy_cal(0.0179, 0.19410)
    s = l.list2spe()
    s.plot_erg_spectrum()

    # p = '/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/user_saved_data/SpecTestingData/Co60_1.Lis'
    # # l = MaestroListFile(p, max_words=None)
    # # l.pickle()
    # _, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
    # l = MaestroListFile.from_pickle(p)
    #
    # spe_true = SPEFile('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/user_saved_data/SpecTestingData/Co60_1.Spe')
    # #
    # spe_from_list = l.SPE
    # spe_true.plot_erg_spectrum(1170, 1176,  ax=ax1, alpha=1,  make_rate=True)
    # spe_from_list.plot_erg_spectrum(1170, 1176, ax=ax1, alpha=1, make_rate=True)
    plt.show()

