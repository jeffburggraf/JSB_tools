from openmc.data.reaction import REACTION_NAME
import re

mt_data = {11: {'level': 0, 'name': '2nd', 'outz': 1, 'outn': 3}, 16: {'level': 0, 'name': '2n', 'outz': 0, 'outn': 2},
           17: {'level': 0, 'name': '3n', 'outz': 0, 'outn': 3}, 22: {'level': 0, 'name': 'na', 'outz': 2, 'outn': 3},
           23: {'level': 0, 'name': 'n3a', 'outz': 6, 'outn': 7}, 24: {'level': 0, 'name': '2na', 'outz': 2, 'outn': 4},
           25: {'level': 0, 'name': '3na', 'outz': 2, 'outn': 5}, 28: {'level': 0, 'name': 'np', 'outz': 1, 'outn': 1},
           29: {'level': 0, 'name': 'n2a', 'outz': 4, 'outn': 5},
           30: {'level': 0, 'name': '2n2a', 'outz': 4, 'outn': 6}, 32: {'level': 0, 'name': 'nd', 'outz': 1, 'outn': 2},
           33: {'level': 0, 'name': 'nt', 'outz': 1, 'outn': 3}, 34: {'level': 0, 'name': 'n3He', 'outz': 2, 'outn': 2},
           35: {'level': 0, 'name': 'nd2a', 'outz': 5, 'outn': 6},
           36: {'level': 0, 'name': 'nt2a', 'outz': 5, 'outn': 7}, 37: {'level': 0, 'name': '4n', 'outz': 0, 'outn': 4},
           41: {'level': 0, 'name': '2np', 'outz': 1, 'outn': 2}, 42: {'level': 0, 'name': '3np', 'outz': 1, 'outn': 3},
           44: {'level': 0, 'name': 'n2p', 'outz': 2, 'outn': 1}, 45: {'level': 0, 'name': 'npa', 'outz': 3, 'outn': 3},
           102: {'level': 0, 'name': 'gamma', 'outn': 0, 'outz': 0},
           103: {'level': 0, 'name': 'p', 'outz': 1, 'outn': 0}, 104: {'level': 0, 'name': 'd', 'outz': 1, 'outn': 1},
           105: {'level': 0, 'name': 't', 'outz': 1, 'outn': 2}, 106: {'level': 0, 'name': '3He', 'outz': 2, 'outn': 1},
           107: {'level': 0, 'name': 'a', 'outz': 2, 'outn': 2}, 108: {'level': 0, 'name': '2a', 'outz': 4, 'outn': 4},
           109: {'level': 0, 'name': '3a', 'outz': 6, 'outn': 6}, 111: {'level': 0, 'name': '2p', 'outz': 2, 'outn': 0},
           112: {'level': 0, 'name': 'pa', 'outz': 3, 'outn': 2},
           113: {'level': 0, 'name': 't2a', 'outz': 5, 'outn': 6},
           114: {'level': 0, 'name': 'd2a', 'outz': 5, 'outn': 5},
           115: {'level': 0, 'name': 'pd', 'outz': 2, 'outn': 1}, 116: {'level': 0, 'name': 'pt', 'outz': 2, 'outn': 2},
           117: {'level': 0, 'name': 'da', 'outz': 3, 'outn': 3}, 152: {'level': 0, 'name': '5n', 'outz': 0, 'outn': 5},
           153: {'level': 0, 'name': '6n', 'outz': 0, 'outn': 6},
           154: {'level': 0, 'name': '2nt', 'outz': 1, 'outn': 4},
           155: {'level': 0, 'name': 'ta', 'outz': 3, 'outn': 4},
           156: {'level': 0, 'name': '4np', 'outz': 1, 'outn': 4},
           157: {'level': 0, 'name': '3nd', 'outz': 1, 'outn': 4},
           158: {'level': 0, 'name': 'nda', 'outz': 3, 'outn': 4},
           159: {'level': 0, 'name': '2npa', 'outz': 3, 'outn': 4},
           160: {'level': 0, 'name': '7n', 'outz': 0, 'outn': 7}, 161: {'level': 0, 'name': '8n', 'outz': 0, 'outn': 8},
           162: {'level': 0, 'name': '5np', 'outz': 1, 'outn': 5},
           163: {'level': 0, 'name': '6np', 'outz': 1, 'outn': 6},
           164: {'level': 0, 'name': '7np', 'outz': 1, 'outn': 7},
           165: {'level': 0, 'name': '4na', 'outz': 2, 'outn': 6},
           166: {'level': 0, 'name': '5na', 'outz': 2, 'outn': 7},
           167: {'level': 0, 'name': '6na', 'outz': 2, 'outn': 8},
           168: {'level': 0, 'name': '7na', 'outz': 2, 'outn': 9},
           169: {'level': 0, 'name': '4nd', 'outz': 1, 'outn': 5},
           170: {'level': 0, 'name': '5nd', 'outz': 1, 'outn': 6},
           171: {'level': 0, 'name': '6nd', 'outz': 1, 'outn': 7},
           172: {'level': 0, 'name': '3nt', 'outz': 1, 'outn': 5},
           173: {'level': 0, 'name': '4nt', 'outz': 1, 'outn': 6},
           174: {'level': 0, 'name': '5nt', 'outz': 1, 'outn': 7},
           175: {'level': 0, 'name': '6nt', 'outz': 1, 'outn': 8},
           176: {'level': 0, 'name': '2n3He', 'outz': 2, 'outn': 3},
           177: {'level': 0, 'name': '3n3He', 'outz': 2, 'outn': 4},
           178: {'level': 0, 'name': '4n3He', 'outz': 2, 'outn': 5},
           179: {'level': 0, 'name': '3n2p', 'outz': 2, 'outn': 3},
           180: {'level': 0, 'name': '3n2a', 'outz': 4, 'outn': 7},
           181: {'level': 0, 'name': '3npa', 'outz': 3, 'outn': 5},
           182: {'level': 0, 'name': 'dt', 'outz': 2, 'outn': 3},
           183: {'level': 0, 'name': 'npd', 'outz': 2, 'outn': 2},
           184: {'level': 0, 'name': 'npt', 'outz': 2, 'outn': 3},
           185: {'level': 0, 'name': 'ndt', 'outz': 2, 'outn': 4},
           186: {'level': 0, 'name': 'np3He', 'outz': 3, 'outn': 2},
           187: {'level': 0, 'name': 'nd3He', 'outz': 3, 'outn': 3},
           188: {'level': 0, 'name': 'nt3He', 'outz': 3, 'outn': 4},
           189: {'level': 0, 'name': 'nta', 'outz': 3, 'outn': 5},
           190: {'level': 0, 'name': '2n2p', 'outz': 2, 'outn': 2},
           191: {'level': 0, 'name': 'p3He', 'outz': 3, 'outn': 1},
           192: {'level': 0, 'name': 'd3He', 'outz': 3, 'outn': 2},
           193: {'level': 0, 'name': '3Hea', 'outz': 4, 'outn': 3},
           194: {'level': 0, 'name': '4n2p', 'outz': 2, 'outn': 4},
           195: {'level': 0, 'name': '4n2a', 'outz': 4, 'outn': 8},
           196: {'level': 0, 'name': '4npa', 'outz': 3, 'outn': 6},
           197: {'level': 0, 'name': '3p', 'outz': 3, 'outn': 0},
           198: {'level': 0, 'name': 'n3p', 'outz': 3, 'outn': 1},
           199: {'level': 0, 'name': '3n2pa', 'outz': 4, 'outn': 5},
           200: {'level': 0, 'name': '5n2p', 'outz': 2, 'outn': 5},
           600: {'level': 0, 'name': 'p0', 'outz': 1, 'outn': 0}, 601: {'level': 1, 'name': 'p1', 'outz': 1, 'outn': 0},
           602: {'level': 2, 'name': 'p2', 'outz': 1, 'outn': 0}, 603: {'level': 3, 'name': 'p3', 'outz': 1, 'outn': 0},
           604: {'level': 4, 'name': 'p4', 'outz': 1, 'outn': 0}, 605: {'level': 5, 'name': 'p5', 'outz': 1, 'outn': 0},
           606: {'level': 6, 'name': 'p6', 'outz': 1, 'outn': 0}, 607: {'level': 7, 'name': 'p7', 'outz': 1, 'outn': 0},
           608: {'level': 8, 'name': 'p8', 'outz': 1, 'outn': 0}, 609: {'level': 9, 'name': 'p9', 'outz': 1, 'outn': 0},
           610: {'level': 10, 'name': 'p10', 'outz': 1, 'outn': 0},
           611: {'level': 11, 'name': 'p11', 'outz': 1, 'outn': 0},
           612: {'level': 12, 'name': 'p12', 'outz': 1, 'outn': 0},
           613: {'level': 13, 'name': 'p13', 'outz': 1, 'outn': 0},
           614: {'level': 14, 'name': 'p14', 'outz': 1, 'outn': 0},
           615: {'level': 15, 'name': 'p15', 'outz': 1, 'outn': 0},
           616: {'level': 16, 'name': 'p16', 'outz': 1, 'outn': 0},
           617: {'level': 17, 'name': 'p17', 'outz': 1, 'outn': 0},
           618: {'level': 18, 'name': 'p18', 'outz': 1, 'outn': 0},
           619: {'level': 19, 'name': 'p19', 'outz': 1, 'outn': 0},
           620: {'level': 20, 'name': 'p20', 'outz': 1, 'outn': 0},
           621: {'level': 21, 'name': 'p21', 'outz': 1, 'outn': 0},
           622: {'level': 22, 'name': 'p22', 'outz': 1, 'outn': 0},
           623: {'level': 23, 'name': 'p23', 'outz': 1, 'outn': 0},
           624: {'level': 24, 'name': 'p24', 'outz': 1, 'outn': 0},
           625: {'level': 25, 'name': 'p25', 'outz': 1, 'outn': 0},
           626: {'level': 26, 'name': 'p26', 'outz': 1, 'outn': 0},
           627: {'level': 27, 'name': 'p27', 'outz': 1, 'outn': 0},
           628: {'level': 28, 'name': 'p28', 'outz': 1, 'outn': 0},
           629: {'level': 29, 'name': 'p29', 'outz': 1, 'outn': 0},
           630: {'level': 30, 'name': 'p30', 'outz': 1, 'outn': 0},
           631: {'level': 31, 'name': 'p31', 'outz': 1, 'outn': 0},
           632: {'level': 32, 'name': 'p32', 'outz': 1, 'outn': 0},
           633: {'level': 33, 'name': 'p33', 'outz': 1, 'outn': 0},
           634: {'level': 34, 'name': 'p34', 'outz': 1, 'outn': 0},
           635: {'level': 35, 'name': 'p35', 'outz': 1, 'outn': 0},
           636: {'level': 36, 'name': 'p36', 'outz': 1, 'outn': 0},
           637: {'level': 37, 'name': 'p37', 'outz': 1, 'outn': 0},
           638: {'level': 38, 'name': 'p38', 'outz': 1, 'outn': 0},
           639: {'level': 39, 'name': 'p39', 'outz': 1, 'outn': 0},
           640: {'level': 40, 'name': 'p40', 'outz': 1, 'outn': 0},
           641: {'level': 41, 'name': 'p41', 'outz': 1, 'outn': 0},
           642: {'level': 42, 'name': 'p42', 'outz': 1, 'outn': 0},
           643: {'level': 43, 'name': 'p43', 'outz': 1, 'outn': 0},
           644: {'level': 44, 'name': 'p44', 'outz': 1, 'outn': 0},
           645: {'level': 45, 'name': 'p45', 'outz': 1, 'outn': 0},
           646: {'level': 46, 'name': 'p46', 'outz': 1, 'outn': 0},
           647: {'level': 47, 'name': 'p47', 'outz': 1, 'outn': 0},
           648: {'level': 48, 'name': 'p48', 'outz': 1, 'outn': 0},
           650: {'level': 0, 'name': 'd0', 'outz': 1, 'outn': 1}, 651: {'level': 1, 'name': 'd1', 'outz': 1, 'outn': 1},
           652: {'level': 2, 'name': 'd2', 'outz': 1, 'outn': 1}, 653: {'level': 3, 'name': 'd3', 'outz': 1, 'outn': 1},
           654: {'level': 4, 'name': 'd4', 'outz': 1, 'outn': 1}, 655: {'level': 5, 'name': 'd5', 'outz': 1, 'outn': 1},
           656: {'level': 6, 'name': 'd6', 'outz': 1, 'outn': 1}, 657: {'level': 7, 'name': 'd7', 'outz': 1, 'outn': 1},
           658: {'level': 8, 'name': 'd8', 'outz': 1, 'outn': 1}, 659: {'level': 9, 'name': 'd9', 'outz': 1, 'outn': 1},
           660: {'level': 10, 'name': 'd10', 'outz': 1, 'outn': 1},
           661: {'level': 11, 'name': 'd11', 'outz': 1, 'outn': 1},
           662: {'level': 12, 'name': 'd12', 'outz': 1, 'outn': 1},
           663: {'level': 13, 'name': 'd13', 'outz': 1, 'outn': 1},
           664: {'level': 14, 'name': 'd14', 'outz': 1, 'outn': 1},
           665: {'level': 15, 'name': 'd15', 'outz': 1, 'outn': 1},
           666: {'level': 16, 'name': 'd16', 'outz': 1, 'outn': 1},
           667: {'level': 17, 'name': 'd17', 'outz': 1, 'outn': 1},
           668: {'level': 18, 'name': 'd18', 'outz': 1, 'outn': 1},
           669: {'level': 19, 'name': 'd19', 'outz': 1, 'outn': 1},
           670: {'level': 20, 'name': 'd20', 'outz': 1, 'outn': 1},
           671: {'level': 21, 'name': 'd21', 'outz': 1, 'outn': 1},
           672: {'level': 22, 'name': 'd22', 'outz': 1, 'outn': 1},
           673: {'level': 23, 'name': 'd23', 'outz': 1, 'outn': 1},
           674: {'level': 24, 'name': 'd24', 'outz': 1, 'outn': 1},
           675: {'level': 25, 'name': 'd25', 'outz': 1, 'outn': 1},
           676: {'level': 26, 'name': 'd26', 'outz': 1, 'outn': 1},
           677: {'level': 27, 'name': 'd27', 'outz': 1, 'outn': 1},
           678: {'level': 28, 'name': 'd28', 'outz': 1, 'outn': 1},
           679: {'level': 29, 'name': 'd29', 'outz': 1, 'outn': 1},
           680: {'level': 30, 'name': 'd30', 'outz': 1, 'outn': 1},
           681: {'level': 31, 'name': 'd31', 'outz': 1, 'outn': 1},
           682: {'level': 32, 'name': 'd32', 'outz': 1, 'outn': 1},
           683: {'level': 33, 'name': 'd33', 'outz': 1, 'outn': 1},
           684: {'level': 34, 'name': 'd34', 'outz': 1, 'outn': 1},
           685: {'level': 35, 'name': 'd35', 'outz': 1, 'outn': 1},
           686: {'level': 36, 'name': 'd36', 'outz': 1, 'outn': 1},
           687: {'level': 37, 'name': 'd37', 'outz': 1, 'outn': 1},
           688: {'level': 38, 'name': 'd38', 'outz': 1, 'outn': 1},
           689: {'level': 39, 'name': 'd39', 'outz': 1, 'outn': 1},
           690: {'level': 40, 'name': 'd40', 'outz': 1, 'outn': 1},
           691: {'level': 41, 'name': 'd41', 'outz': 1, 'outn': 1},
           692: {'level': 42, 'name': 'd42', 'outz': 1, 'outn': 1},
           693: {'level': 43, 'name': 'd43', 'outz': 1, 'outn': 1},
           694: {'level': 44, 'name': 'd44', 'outz': 1, 'outn': 1},
           695: {'level': 45, 'name': 'd45', 'outz': 1, 'outn': 1},
           696: {'level': 46, 'name': 'd46', 'outz': 1, 'outn': 1},
           697: {'level': 47, 'name': 'd47', 'outz': 1, 'outn': 1},
           698: {'level': 48, 'name': 'd48', 'outz': 1, 'outn': 1},
           700: {'level': 0, 'name': 't0', 'outz': 1, 'outn': 2}, 701: {'level': 1, 'name': 't1', 'outz': 1, 'outn': 2},
           702: {'level': 2, 'name': 't2', 'outz': 1, 'outn': 2}, 703: {'level': 3, 'name': 't3', 'outz': 1, 'outn': 2},
           704: {'level': 4, 'name': 't4', 'outz': 1, 'outn': 2}, 705: {'level': 5, 'name': 't5', 'outz': 1, 'outn': 2},
           706: {'level': 6, 'name': 't6', 'outz': 1, 'outn': 2}, 707: {'level': 7, 'name': 't7', 'outz': 1, 'outn': 2},
           708: {'level': 8, 'name': 't8', 'outz': 1, 'outn': 2}, 709: {'level': 9, 'name': 't9', 'outz': 1, 'outn': 2},
           710: {'level': 10, 'name': 't10', 'outz': 1, 'outn': 2},
           711: {'level': 11, 'name': 't11', 'outz': 1, 'outn': 2},
           712: {'level': 12, 'name': 't12', 'outz': 1, 'outn': 2},
           713: {'level': 13, 'name': 't13', 'outz': 1, 'outn': 2},
           714: {'level': 14, 'name': 't14', 'outz': 1, 'outn': 2},
           715: {'level': 15, 'name': 't15', 'outz': 1, 'outn': 2},
           716: {'level': 16, 'name': 't16', 'outz': 1, 'outn': 2},
           717: {'level': 17, 'name': 't17', 'outz': 1, 'outn': 2},
           718: {'level': 18, 'name': 't18', 'outz': 1, 'outn': 2},
           719: {'level': 19, 'name': 't19', 'outz': 1, 'outn': 2},
           720: {'level': 20, 'name': 't20', 'outz': 1, 'outn': 2},
           721: {'level': 21, 'name': 't21', 'outz': 1, 'outn': 2},
           722: {'level': 22, 'name': 't22', 'outz': 1, 'outn': 2},
           723: {'level': 23, 'name': 't23', 'outz': 1, 'outn': 2},
           724: {'level': 24, 'name': 't24', 'outz': 1, 'outn': 2},
           725: {'level': 25, 'name': 't25', 'outz': 1, 'outn': 2},
           726: {'level': 26, 'name': 't26', 'outz': 1, 'outn': 2},
           727: {'level': 27, 'name': 't27', 'outz': 1, 'outn': 2},
           728: {'level': 28, 'name': 't28', 'outz': 1, 'outn': 2},
           729: {'level': 29, 'name': 't29', 'outz': 1, 'outn': 2},
           730: {'level': 30, 'name': 't30', 'outz': 1, 'outn': 2},
           731: {'level': 31, 'name': 't31', 'outz': 1, 'outn': 2},
           732: {'level': 32, 'name': 't32', 'outz': 1, 'outn': 2},
           733: {'level': 33, 'name': 't33', 'outz': 1, 'outn': 2},
           734: {'level': 34, 'name': 't34', 'outz': 1, 'outn': 2},
           735: {'level': 35, 'name': 't35', 'outz': 1, 'outn': 2},
           736: {'level': 36, 'name': 't36', 'outz': 1, 'outn': 2},
           737: {'level': 37, 'name': 't37', 'outz': 1, 'outn': 2},
           738: {'level': 38, 'name': 't38', 'outz': 1, 'outn': 2},
           739: {'level': 39, 'name': 't39', 'outz': 1, 'outn': 2},
           740: {'level': 40, 'name': 't40', 'outz': 1, 'outn': 2},
           741: {'level': 41, 'name': 't41', 'outz': 1, 'outn': 2},
           742: {'level': 42, 'name': 't42', 'outz': 1, 'outn': 2},
           743: {'level': 43, 'name': 't43', 'outz': 1, 'outn': 2},
           744: {'level': 44, 'name': 't44', 'outz': 1, 'outn': 2},
           745: {'level': 45, 'name': 't45', 'outz': 1, 'outn': 2},
           746: {'level': 46, 'name': 't46', 'outz': 1, 'outn': 2},
           747: {'level': 47, 'name': 't47', 'outz': 1, 'outn': 2},
           748: {'level': 48, 'name': 't48', 'outz': 1, 'outn': 2},
           750: {'level': 0, 'name': '3He0', 'outz': 2, 'outn': 1},
           751: {'level': 1, 'name': '3He1', 'outz': 2, 'outn': 1},
           752: {'level': 2, 'name': '3He2', 'outz': 2, 'outn': 1},
           753: {'level': 3, 'name': '3He3', 'outz': 2, 'outn': 1},
           754: {'level': 4, 'name': '3He4', 'outz': 2, 'outn': 1},
           755: {'level': 5, 'name': '3He5', 'outz': 2, 'outn': 1},
           756: {'level': 6, 'name': '3He6', 'outz': 2, 'outn': 1},
           757: {'level': 7, 'name': '3He7', 'outz': 2, 'outn': 1},
           758: {'level': 8, 'name': '3He8', 'outz': 2, 'outn': 1},
           759: {'level': 9, 'name': '3He9', 'outz': 2, 'outn': 1},
           760: {'level': 10, 'name': '3He10', 'outz': 2, 'outn': 1},
           761: {'level': 11, 'name': '3He11', 'outz': 2, 'outn': 1},
           762: {'level': 12, 'name': '3He12', 'outz': 2, 'outn': 1},
           763: {'level': 13, 'name': '3He13', 'outz': 2, 'outn': 1},
           764: {'level': 14, 'name': '3He14', 'outz': 2, 'outn': 1},
           765: {'level': 15, 'name': '3He15', 'outz': 2, 'outn': 1},
           766: {'level': 16, 'name': '3He16', 'outz': 2, 'outn': 1},
           767: {'level': 17, 'name': '3He17', 'outz': 2, 'outn': 1},
           768: {'level': 18, 'name': '3He18', 'outz': 2, 'outn': 1},
           769: {'level': 19, 'name': '3He19', 'outz': 2, 'outn': 1},
           770: {'level': 20, 'name': '3He20', 'outz': 2, 'outn': 1},
           771: {'level': 21, 'name': '3He21', 'outz': 2, 'outn': 1},
           772: {'level': 22, 'name': '3He22', 'outz': 2, 'outn': 1},
           773: {'level': 23, 'name': '3He23', 'outz': 2, 'outn': 1},
           774: {'level': 24, 'name': '3He24', 'outz': 2, 'outn': 1},
           775: {'level': 25, 'name': '3He25', 'outz': 2, 'outn': 1},
           776: {'level': 26, 'name': '3He26', 'outz': 2, 'outn': 1},
           777: {'level': 27, 'name': '3He27', 'outz': 2, 'outn': 1},
           778: {'level': 28, 'name': '3He28', 'outz': 2, 'outn': 1},
           779: {'level': 29, 'name': '3He29', 'outz': 2, 'outn': 1},
           780: {'level': 30, 'name': '3He30', 'outz': 2, 'outn': 1},
           781: {'level': 31, 'name': '3He31', 'outz': 2, 'outn': 1},
           782: {'level': 32, 'name': '3He32', 'outz': 2, 'outn': 1},
           783: {'level': 33, 'name': '3He33', 'outz': 2, 'outn': 1},
           784: {'level': 34, 'name': '3He34', 'outz': 2, 'outn': 1},
           785: {'level': 35, 'name': '3He35', 'outz': 2, 'outn': 1},
           786: {'level': 36, 'name': '3He36', 'outz': 2, 'outn': 1},
           787: {'level': 37, 'name': '3He37', 'outz': 2, 'outn': 1},
           788: {'level': 38, 'name': '3He38', 'outz': 2, 'outn': 1},
           789: {'level': 39, 'name': '3He39', 'outz': 2, 'outn': 1},
           790: {'level': 40, 'name': '3He40', 'outz': 2, 'outn': 1},
           791: {'level': 41, 'name': '3He41', 'outz': 2, 'outn': 1},
           792: {'level': 42, 'name': '3He42', 'outz': 2, 'outn': 1},
           793: {'level': 43, 'name': '3He43', 'outz': 2, 'outn': 1},
           794: {'level': 44, 'name': '3He44', 'outz': 2, 'outn': 1},
           795: {'level': 45, 'name': '3He45', 'outz': 2, 'outn': 1},
           796: {'level': 46, 'name': '3He46', 'outz': 2, 'outn': 1},
           797: {'level': 47, 'name': '3He47', 'outz': 2, 'outn': 1},
           798: {'level': 48, 'name': '3He48', 'outz': 2, 'outn': 1},
           800: {'level': 0, 'name': 'a0', 'outz': 2, 'outn': 2}, 801: {'level': 1, 'name': 'a1', 'outz': 2, 'outn': 2},
           802: {'level': 2, 'name': 'a2', 'outz': 2, 'outn': 2}, 803: {'level': 3, 'name': 'a3', 'outz': 2, 'outn': 2},
           804: {'level': 4, 'name': 'a4', 'outz': 2, 'outn': 2}, 805: {'level': 5, 'name': 'a5', 'outz': 2, 'outn': 2},
           806: {'level': 6, 'name': 'a6', 'outz': 2, 'outn': 2}, 807: {'level': 7, 'name': 'a7', 'outz': 2, 'outn': 2},
           808: {'level': 8, 'name': 'a8', 'outz': 2, 'outn': 2}, 809: {'level': 9, 'name': 'a9', 'outz': 2, 'outn': 2},
           810: {'level': 10, 'name': 'a10', 'outz': 2, 'outn': 2},
           811: {'level': 11, 'name': 'a11', 'outz': 2, 'outn': 2},
           812: {'level': 12, 'name': 'a12', 'outz': 2, 'outn': 2},
           813: {'level': 13, 'name': 'a13', 'outz': 2, 'outn': 2},
           814: {'level': 14, 'name': 'a14', 'outz': 2, 'outn': 2},
           815: {'level': 15, 'name': 'a15', 'outz': 2, 'outn': 2},
           816: {'level': 16, 'name': 'a16', 'outz': 2, 'outn': 2},
           817: {'level': 17, 'name': 'a17', 'outz': 2, 'outn': 2},
           818: {'level': 18, 'name': 'a18', 'outz': 2, 'outn': 2},
           819: {'level': 19, 'name': 'a19', 'outz': 2, 'outn': 2},
           820: {'level': 20, 'name': 'a20', 'outz': 2, 'outn': 2},
           821: {'level': 21, 'name': 'a21', 'outz': 2, 'outn': 2},
           822: {'level': 22, 'name': 'a22', 'outz': 2, 'outn': 2},
           823: {'level': 23, 'name': 'a23', 'outz': 2, 'outn': 2},
           824: {'level': 24, 'name': 'a24', 'outz': 2, 'outn': 2},
           825: {'level': 25, 'name': 'a25', 'outz': 2, 'outn': 2},
           826: {'level': 26, 'name': 'a26', 'outz': 2, 'outn': 2},
           827: {'level': 27, 'name': 'a27', 'outz': 2, 'outn': 2},
           828: {'level': 28, 'name': 'a28', 'outz': 2, 'outn': 2},
           829: {'level': 29, 'name': 'a29', 'outz': 2, 'outn': 2},
           830: {'level': 30, 'name': 'a30', 'outz': 2, 'outn': 2},
           831: {'level': 31, 'name': 'a31', 'outz': 2, 'outn': 2},
           832: {'level': 32, 'name': 'a32', 'outz': 2, 'outn': 2},
           833: {'level': 33, 'name': 'a33', 'outz': 2, 'outn': 2},
           834: {'level': 34, 'name': 'a34', 'outz': 2, 'outn': 2},
           835: {'level': 35, 'name': 'a35', 'outz': 2, 'outn': 2},
           836: {'level': 36, 'name': 'a36', 'outz': 2, 'outn': 2},
           837: {'level': 37, 'name': 'a37', 'outz': 2, 'outn': 2},
           838: {'level': 38, 'name': 'a38', 'outz': 2, 'outn': 2},
           839: {'level': 39, 'name': 'a39', 'outz': 2, 'outn': 2},
           840: {'level': 40, 'name': 'a40', 'outz': 2, 'outn': 2},
           841: {'level': 41, 'name': 'a41', 'outz': 2, 'outn': 2},
           842: {'level': 42, 'name': 'a42', 'outz': 2, 'outn': 2},
           843: {'level': 43, 'name': 'a43', 'outz': 2, 'outn': 2},
           844: {'level': 44, 'name': 'a44', 'outz': 2, 'outn': 2},
           845: {'level': 45, 'name': 'a45', 'outz': 2, 'outn': 2},
           846: {'level': 46, 'name': 'a46', 'outz': 2, 'outn': 2},
           847: {'level': 47, 'name': 'a47', 'outz': 2, 'outn': 2},
           848: {'level': 48, 'name': 'a48', 'outz': 2, 'outn': 2},
           875: {'level': 0, 'name': '2n0', 'outz': 0, 'outn': 2},
           876: {'level': 1, 'name': '2n1', 'outz': 0, 'outn': 2},
           877: {'level': 2, 'name': '2n2', 'outz': 0, 'outn': 2},
           878: {'level': 3, 'name': '2n3', 'outz': 0, 'outn': 2},
           879: {'level': 4, 'name': '2n4', 'outz': 0, 'outn': 2},
           880: {'level': 5, 'name': '2n5', 'outz': 0, 'outn': 2},
           881: {'level': 6, 'name': '2n6', 'outz': 0, 'outn': 2},
           882: {'level': 7, 'name': '2n7', 'outz': 0, 'outn': 2},
           883: {'level': 8, 'name': '2n8', 'outz': 0, 'outn': 2},
           884: {'level': 9, 'name': '2n9', 'outz': 0, 'outn': 2},
           885: {'level': 10, 'name': '2n10', 'outz': 0, 'outn': 2},
           886: {'level': 11, 'name': '2n11', 'outz': 0, 'outn': 2},
           887: {'level': 12, 'name': '2n12', 'outz': 0, 'outn': 2},
           888: {'level': 13, 'name': '2n13', 'outz': 0, 'outn': 2},
           889: {'level': 14, 'name': '2n14', 'outz': 0, 'outn': 2},
           890: {'level': 15, 'name': '2n15', 'outz': 0, 'outn': 2}}


if __name__ == '__main__':
    z_n = {'a': (2, 2), 't': (1, 2), 'p':(1, 0), 'd': (1, 1), 'n': (0, 1), '3He': (2, 1)}
    data = {}

    for mt, v in REACTION_NAME.items():
        m = re.match('\((.),(.+)\)', v)
        # print(f'MT={mt} {v}')
        # data[mt] = {'level': 0, 'outn': None, 'outz': None}

        if m is not None:
            data[mt] = {'level': 0}
            in_ = m.groups()[0]
            outs = m.groups()[1]
            data[mt]['name'] = outs
            if outs == 'gamma':
                data[mt]['outn'] = data[mt]['outz'] = 0
                continue
            if ('f' in outs or 'X' in outs or 'c' in outs) or outs in ['disappear', 'absorption', 'elastic', 'total', 'level', 'misc', 'fission']:
                del data[mt]
                continue
            l_match = re.match('.+?([0-9]+)$', outs)
            if l_match:
                l = int(l_match.groups()[0])
                outs = outs[:l_match.start(1)]
            else:
                l = 0

            data[mt]['level'] = l
            outz, outn = 0, 0

            for out in re.finditer('(?:([0-9]?)([nptda]|3He))', outs):
                if out.groups()[0]:
                    npar = int(out.groups()[0])
                else:
                    npar = 1

                par = out.groups()[1]

                outz += npar * z_n[par][0]
                outn += npar * z_n[par][1]

            data[mt]['outz'] = outz
            data[mt]['outn'] = outn

    for k, v in data.items():
        print(k, v)

    print(data)