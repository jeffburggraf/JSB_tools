import struct
from struct import unpack, calcsize
from pathlib import Path
import datetime
from bitstring import BitStream
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from datetime import timezone
from dateutil import tz
import filetime
HERE = tz.tzlocal()
OLE_TIME_ZERO = datetime.datetime(1899, 12, 30, 0, 0, 0)

EPOCH_AS_FILETIME = 116444736000000000  # January 1, 1970 as MS file time
HUNDREDS_OF_NANOSECONDS = 10000000


def filetime_to_dt(ft):
    """Converts a Microsoft filetime number to a Python datetime. The new
    datetime object is time zone-naive but is equivalent to tzinfo=utc.
    >>> filetime_to_dt(116444736000000000)
    datetime.datetime(1970, 1, 1, 0, 0)
    >>> filetime_to_dt(128930364000000000)
    datetime.datetime(2009, 7, 25, 23, 0)
    """
    return datetime.datetime((ft - EPOCH_AS_FILETIME) / HUNDREDS_OF_NANOSECONDS)


def ole2datetime(oledt):
    return OLE_TIME_ZERO + datetime.timedelta(days=float(oledt))


class ListFile:
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
        self.index += 4
        return out

    def process_32(self, debug=False):
        """
        Details of Clock Rollover:
        Every 10ms the number of 200ns clock ticks is reset to zero. Determining when this happens is not straightforward.
        The manual is incorrect when it claims that there is an RT word every time the clock rolls over.
        There is one SOMETIMES. However, whenever there is not an RT word during a clock rollover, there appears to be
        an LT word. So both LT and RT words are used to increment the number of rollovers.
        When RT words do appear, they are consistently equal to the # of clock rollovers, so the value of
        self.n_rollovers is tethered to the value of each RT word when it appears, but to correct for missing RT words,
        self.n_rollovers is incremented in the presence of an LT word.
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
            real_time = self.n_rollovers * 10E-3 + n_200ns_ticks * 200E-9
            self.times.append(real_time)
            if debug:
                word_debug = f"ADC word, Value = {adc_value}, n_200ns_ticks = {n_200ns_ticks}, real time: {real_time}," \
                             f"n rollovers: {self.n_rollovers}"
            self.dont_roll_over = False

        elif word[:2] == "01":  # LT word
            live_time_10ms = int(word[2:], 2)
            if debug:
                word_debug = f"LT word: {live_time_10ms}"
            # self.dont_roll_over is to prevent from incrementing again on RT word in case RT & LT ever occur together
            # (I haven't seen this happen yet)
            if not self.dont_roll_over:
                self.n_rollovers += 1
            if len(self.times):
                self.realtimes.append(self.times[-1])
            else:
                self.realtimes.append(0)
            self.livetimes.append(10E-3*live_time_10ms)

        elif word[:2] == "10":  # RT word
            real_time_10ms = int(word[2:], 2)
            if debug:
                word_debug = f"RT word: {real_time_10ms}"
            self.n_rollovers = real_time_10ms
            self.dont_roll_over = True

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

        elif word[:6] == '000000':
            if word[:8] == '00000001':
                # self.wintime = [word[-8:], word[16:24],word[8:16],0,0,0,0,0]
                self.wintime[0] = word[-8:]
                self.wintime[1] = word[-16:-8]
                self.wintime[2] = word[8:16]
                if debug:
                    word_debug = f"Connections Time (1): {word[:8], word[8:16], word[16:24], word[24:]}"

            elif word[:8] == '00000010':
                if debug:
                    if debug:
                        word_debug = f"Connections Time (2): {word[:8], word[8:16], word[16:24], word[24:]}"
                    self.wintime[3] = word[-8:]
                    self.wintime[4] = word[-16:-8]
                    self.wintime[5] = word[8:16]
            elif word[:8] == '00000011':
                if debug:
                    word_debug = f"Connections Time (3): {word[:8], word[8:16], word[16:24], word[24:]}"
                    self.wintime[6] = word[-8:]
                    self.wintime[7] = word[-16:-8]
            else:
                assert False, f"Unexpected word: {word}"

            if all([a is not None for a in self.wintime]):
                self.wintime.reverse()
                # high = "".join(self.wintime[:4])
                # high = high
                w_time = "".join(self.wintime)
                w_time = BitStream(bin=w_time).unpack('uintbe:64')[0]
                w_time = filetime.to_datetime(w_time)
                # high = BitStream(bin=w_time[:len(w_time)//2])
                # low = BitStream(bin=w_time[len(w_time)//2:])
                # high = w_time[len(w_time)//2]
                # low = w_time[len(w_time)//2:]
                # w_time = high.unpack('uintbe:32')[0]*2**32 + low.unpack('uintbe:32')[0]
                # w_time = w_time.unpack('uintbe:64')[0]

                # low = low[::-1]
                # high = int(high, 2)*2**32
                # print(high)
                # low = int(low, 2)
                # print(low)
                # w_time = filetime_to_dt(w_time)
                # w_time = datetime.datetime(1601, 1,1) + datetime.timedelta(microseconds=w_time/10)
                # w_time = w_time.astimezone(HERE)
                print(f"WinTime: {w_time.astimezone(HERE)}")


                self.wintime = [None] * 8

        else:
            if debug:
                word_debug = f"Unknown word: {word}"

        if debug:
            print(word_debug)
        return True

    def __init__(self, path, max_events=None, debug=False):
        path = Path(path)
        assert path.exists()
        with open(path, 'rb') as f:
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
        self.read('9s')

        self.n_rollovers = 0  # Everytime the clock rolls over ()every 10 ms),
        # we see an RT word indicating the number of roll overs.
        self.dont_roll_over = False
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

        # convolve to remove defects. Only deadtime changes on the order of 0.1 seconds matter anyways.
        kernel = norm(loc=len(self.livetimes)//2, scale=10).pdf(np.arange(len(self.livetimes)))  # scale: 0.1 seconds
        delta_realtimes = np.convolve(delta_realtimes, kernel, mode='same')
        delta_livetimes = np.convolve(delta_livetimes, kernel, mode='same')
        self.percent_live = delta_livetimes/delta_realtimes
        plt.plot(np.arange(len(self.livetimes)), delta_realtimes)
        plt.plot(np.arange(len(self.livetimes)), delta_livetimes)
        plt.figure()
        plt.plot(self.realtimes, self.percent_live)
        plt.figure()
        plt.hist(self.percent_live, bins=40)
        # plt.hist(self.energies, bins=600)




# l = ListFile('/Users/burggraf1/Desktop/HPGE_temp/Eu152_SampleIn.Lis',  max_events=10000, debug=True)
l = ListFile('/Users/burggraf1/Desktop/HPGE_temp/firstTest.Lis', max_events=2000, debug=True)
print(l.serial_number)
# 2021-07-15 14:12:13.000

plt.show()
# WinTime: ['01110000', '11011110', '01011001', '11000010', '10110101', '01111001', '11010111', '00000001']
# 132708535574930000
# 132708535587430000
# 132708535592430000  2021-07-14 11:26:19.000
#  2021-07-15 14:12:13.000
#  20:12:16.420000-06:00



#  True 11:26:19.000