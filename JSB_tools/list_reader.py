import struct
from struct import unpack, calcsize
from pathlib import Path
import datetime
from bitstream import BitStream

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
OLE_TIME_ZERO = datetime.datetime(1899, 12, 30, 0, 0, 0)


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
                word_debug = f"Hardware time: {hrd_time}"

        elif word[:8] == '00000100':
            adc_crm = int(word[16:], 2)
            # self.n_rollovers += 1
            if debug:
                word_debug = f"ADC CRM: {adc_crm}"

        elif word[:8] == '00000101':
            counter_1 = int(word[16:], 2)
            if debug:
                word_debug = f"counter 1: {counter_1}"

        elif word[:8] == '00000110':
            counter_2 = int(word[16:], 2)
            if debug:
                word_debug = f"counter 2: {counter_2}"

        elif word[:8] == '00000111':
            gm_counter = int(word[16:], 2)
            if debug:
                word_debug = f"gm counter: {gm_counter}"
        # elif word[:8] == '00000001':
        #     self.wintime = [word[-8:],word[16:24],word[8:16],0,0,0,0,0]
        #
        # elif self.wintime:
        #     if word[:8] == '00000010':
        #         self.wintime[3] = word[-8:]
        #         self.wintime[4] = word[16:24]
        #         self.wintime[5] = word[8:16]
        #     if word[:8] == '00000011':
        #         self.wintime[7] = word[16:24]
        #         self.wintime[6] = word[-8:]
        #     print('WINTIME: ', self.wintime)
        #     self.wintime = None

        else:
            if debug:
                word_debug = f"Unknown word: {word}"

        if debug:
            print(word_debug)
        return True

    def __init__(self, path, debug=False):
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
        self.wintime = None

        self.livetimes = []
        self.realtimes = []

        self.times = []
        while self.process_32(debug):
            pass

        if debug:
            print("Live time: ", self.total_live_time)
            print("Real time: ", self.total_real_time)
            print("Total % live: ", self.total_live_time/self.total_real_time)

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
        print(np.mean(self.percent_live))
        plt.hist(self.percent_live, bins=40)
        # plt.hist(self.energies, bins=600)
        plt.show()




# l = ListFile('/Users/burggraf1/Desktop/HPGE_temp/Eu152_SampleIn.Lis', True)
# l = ListFile(, True)
#1000010111010011001000000 print(bin(10))

with open('/Users/jeffreyburggraf/Desktop/Eu152_SampleIn.Lis', 'rb') as f:
    s = BitStream(f.read(255))
    print(s.read())
