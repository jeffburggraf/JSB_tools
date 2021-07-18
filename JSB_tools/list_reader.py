from __future__ import annotations
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
import binascii
from functools import cached_property
HERE = pytz.timezone('US/Mountain')
OLE_TIME_ZERO = datetime.datetime(1899, 12, 30, 0, 0, 0)
from JSB_tools import ProgressReport
import pickle


def ole2datetime(oledt):
    return OLE_TIME_ZERO + datetime.timedelta(days=float(oledt))


class ListFile:
    datetime_format = '%Y/%m/%d %H:%M:%S.%f (%Z%z)'
    file_datetime_format = '%Y-%m-%d_%H-%M-%S.%f--%Z%z'
    """
    Start time is determined by the Connections Win_64 time sent right after the header. This clock doesn't
    necessarily start with the 10ms rolling-over clock, so the accuracy of the start time is +/- 5ms.
    The SAMPLE READY and GATE counters are also +/- 5 ms.
    Relative times between ADC events are supposedly +/- 200ns. I will need to confirm this.
    """
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
            if debug:
                word_debug = f"ADC word, Value = {adc_value}, n_200ns_ticks = {n_200ns_ticks}, real time: {adc_time}," \
                             f"n rollovers: {self.n_rollovers}"

        elif word[:2] == "01":  # LiveTime word
            live_time_10ms = int(word[2:], 2)
            self.livetimes.append(10E-3*live_time_10ms)
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
            counter_1 = int(word[16:], 2)
            if debug:
                word_debug = f"Counter A (Sample Ready): {counter_1}"

        elif word[:8] == '00000110':
            counter_2 = int(word[16:], 2)
            if debug:
                word_debug = f"Counter B (Gate): {counter_2}"

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

    def __init__(self, path, max_words=None, debug=False):
        path = Path(path)
        assert path.exists()

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

            self.times = []
            self.n_words = 0
            while self.process_32(f, debug=debug):
                if max_words is not None and self.n_words > max_words:
                    break
                self.n_words += 1
        if debug:
            print("Start time: ", self.start_time)
            print("Live time: ", self.total_live_time)
            print("Real time: ", self.total_real_time)
            print("Total % live: ", self.total_live_time/self.total_real_time)
            print("N events/s = ", len(self.times) / self.total_real_time)

        self.adc_values = np.array(self.adc_values)
        self.energies = self.channel_to_erg(self.adc_values)

        # convolve to remove defects. Only deadtime changes on the order of 0.1 seconds matter anyways.
        kernel = norm(loc=len(self.livetimes)//2, scale=10).pdf(np.arange(len(self.livetimes)))  # scale: 0.1 seconds
        self.percent_live = np.gradient(self.livetimes)/np.gradient(self.realtimes)
        self.percent_live = np.convolve(self.percent_live, kernel, mode='same')

        if debug:

            plt.plot(self.realtimes, self.livetimes)
            plt.xlabel("Real-time [s]")
            plt.ylabel("Live-time [s]")
            plt.figure()

            plt.plot(self.realtimes, self.percent_live)
            plt.title("")
            plt.xlabel("Real-time [s]")
            plt.ylabel("% live-time")

            percent_live_hist = TH1F.from_raw_data(self.percent_live, bins=100)
            ax = percent_live_hist.plot(show_stats=True, xlabel="% live-time", ylabel="Counts")
            ax.set_title("Frequencies of percent live-time")

    def channel_to_erg(self, a):
        return np.sum([a ** i * c for i, c in enumerate(self.erg_calibration)], axis=0)

    @cached_property
    def erg_bins(self):
        channel_bins = np.arange(self.n_adc_channels) - 0.5
        return self.channel_to_erg(channel_bins)

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

    @staticmethod
    def __get_auto_data_dir():
        directory = Path(__file__).parent / 'user_saved_data' / 'list_data'
        if not directory.exists():
            directory.mkdir()
        return directory

    def pickle(self, fname=None, directory=None):
        if directory is None:
            directory = self.__get_auto_data_dir()
        if fname is None:
            fname = self.adc_zero_datetime.isoformat(timespec='milliseconds')
        f_path = directory/fname
        with open(f_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fname, directory=None) -> ListFile:
        if directory is None:
            directory = cls.__get_auto_data_dir()
        fpath = directory/fname
        assert fpath.exists(), f'Path does not exist, "{fpath}"'
        with open(fpath, 'rb') as f:
            return pickle.load(f)


import time
t0 = time.time()
l = ListFile('/Users/burggraf1/Desktop/HPGE_temp/Eu152_SampleIn.Lis', max_words=None, debug=False)
# l = ListFile('/Users/burggraf1/Desktop/HPGE_temp/firstTest.Lis', max_words=None, debug=False)
print(time.time() - t0, 'Seconds')

t0 = time.time()
l.pickle('test')
print(time.time() - t0, ' pickle Seconds')

t0 = time.time()
l2 = l.from_pickle('test')
print(time.time() - t0, 'From pickle Seconds')

t0 = time.time()
h = l2.get_spectrum_hist()
print(time.time() - t0, 'get_spectrum_hist Seconds')

t0 = time.time()
h.plot()
print(time.time() - t0, 'PLot seconds')

from JSB_tools import Nuclide

for g in Nuclide.from_symbol('Eu152').decay_gamma_lines:
    print(g)
plt.show()
