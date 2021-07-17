import struct
from struct import unpack, calcsize
from pathlib import Path
import datetime

import pytz
from bitstring import BitStream
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from datetime import timezone
from dateutil import tz
from pytz import common_timezones
# print(common_timezones)
import filetime
HERE = pytz.timezone('US/Mountain')
print(HERE )
OLE_TIME_ZERO = datetime.datetime(1899, 12, 30, 0, 0, 0)



def ole2datetime(oledt):
    return OLE_TIME_ZERO + datetime.timedelta(days=float(oledt))


class ListFile:
    """
    Start time is determined by the Connections Win_64 time sent right after the header. This clock doesn't
    necessarily start with the 10ms rolling-over clock, so the accuracy of the start time is +/- 5ms.
    The SAMPLE READY and GATE counters are also +/- 5 ms.
    Relative times between ADC events are supposedly +/- 200ns. I will need to confirm this.
    """
    def read(self, byte_format, debug=False):
        s = calcsize(byte_format)
        _bytes = self.bytes[self.index: self.index + s]
        self.index += s
        out = unpack(byte_format, _bytes)
        if debug:
            print(f"Byte format: {byte_format}")
            print(f"Calc'd size: {s}")
            print(f"out: {out}")
        if len(out) == 1:
            if byte_format[-1] == 's':
                return out[0].rstrip(b'\x00').decode("windows-1252")
            return out[0]
        else:

            return out

    def read_32(self):
        out = f"{self.read('I'):032b}"
        return out

    def process_32(self, debug=False):
        """
        Args:
            debug:

        Returns:

        """
        try:
            word = self.read_32()
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

        elif word[:2] == "01":  # LT word
            live_time_10ms = int(word[2:], 2)
            self.livetimes.append(10E-3*live_time_10ms)
            if debug:
                word_debug = f"LT word: {live_time_10ms}"

        elif word[:2] == "10":  # RT word
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
            word2 = self.read_32()
            word3 = self.read_32()
            wintime = [0,0,0,0,0,0,0,0]
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

            if self.adc_zero_datetime is None:
                self.adc_zero_datetime = w_time

            if debug:
                word_debug = f"WinTime: {w_time.strftime('%Y-%m-%d %H:%M:%S.%f %Z%z')}"
        else:
            if debug:
                word_debug = f"Unknown word: {word}"

        if debug:
            print(word_debug)
        return True

    def __init__(self, path, max_events=None, debug=False):
        path = Path(path)
        assert path.exists()
        with open(path, 'rb') as f:  # Todo: Don't read this all at once?
            self.bytes = f.read()
        self.index = 0
        lst_header = self.read('i')
        self.adc_values = []
        assert lst_header == -13, f"Invalid list mode header. Should be '-13', got " \
                                                             f"'{lst_header}' instead."
        self.list_style = self.read('i')
        if self.list_style != 2:
            raise NotImplementedError("Digibase and other list styles not yet implemented.")
        self.start_time = ole2datetime(self.read('d')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        self.device_address = self.read("80s")
        self.MCB_type_string = self.read("9s")
        self.serial_number = self.read("16s")
        self.description = self.read('80s')
        self.valid_erg_calibration = self.read('1s')
        self.erg_units = self.read("4s")
        self.erg_calibration = [self.read('f') for i in range(3)]
        self.valid_shape_calibration = self.read('1s')
        self.shape_calibration = [self.read('f') for i in range(3)]
        self.optional_conversion_gain = self.read('i')
        self.det_id_number = self.read('i')
        self.total_real_time = self.read('f')
        self.total_live_time = self.read('f')
        self.adc_zero_datetime = None
        self.read('9s')

        self.n_rollovers = 0  # Everytime the clock rolls over ()every 10 ms),
        # we see an RT word indicating the number of roll overs.
        self.wintime = [None] * 8

        self.livetimes = []
        self.realtimes = []

        self.times = []

        i=0
        while self.process_32(debug):
            if max_events is not None and i > max_events:
                break
            i += 1

        if debug:
            print("Start time: ", self.start_time)
            print("Live time: ", self.total_live_time)
            print("Real time: ", self.total_real_time)
            print("Total % live: ", self.total_live_time/self.total_real_time)
            print("N events/s = ", len(self.times) / self.total_real_time)

        self.adc_values = np.array(self.adc_values)
        self.energies = np.sum([self.adc_values**i*c for i, c in enumerate(self.erg_calibration)], axis=0)

        #  Fraction of live time = dTl/dTr, where Tl is total live time and Tr is total real time (as a function of t).
        delta_realtimes = np.gradient(self.realtimes)
        delta_livetimes = np.gradient(self.livetimes)
        # print(len(self.livetimes), len(self.realtimes))
        # assert False

        # convolve to remove defects. Only deadtime changes on the order of 0.1 seconds matter anyways.
        kernel = norm(loc=len(self.livetimes)//2, scale=10).pdf(np.arange(len(self.livetimes)))  # scale: 0.1 seconds
        # delta_realtimes = np.convolve(delta_realtimes, kernel, mode='same')
        # delta_livetimes = np.convolve(delta_livetimes, kernel, mode='same')
        self.percent_live = np.gradient(self.livetimes)/np.gradient(self.realtimes)
        # plt.plot(np.arange(len(self.livetimes)), delta_realtimes)
        # plt.plot(np.arange(len(self.livetimes)), delta_livetimes)
        l = np.gradient(self.livetimes)/np.gradient(self.realtimes)
        l = np.convolve(l, kernel, mode='same')
        plt.plot(self.realtimes, l)
        plt.figure()
        plt.plot(self.realtimes, self.percent_live)
        plt.figure()
        plt.hist(self.percent_live, bins=40)
        print(self.adc_zero_datetime+ datetime.timedelta(seconds=self.times[-1]))




l = ListFile('/Users/burggraf1/Desktop/HPGE_temp/Eu152_SampleIn.Lis',  max_events=1000, debug=True)
# l = ListFile('/Users/burggraf1/Desktop/HPGE_temp/firstTest.Lis', max_events=None, debug=True)

# plt.show()
