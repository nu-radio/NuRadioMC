# a python list where the keys are the number of a station and the values are the station name
SId_to_Sname = [None] * 209  # just to pre-initilize list, so syntax below is possible
SId_to_Sname[1] = "CS001"
SId_to_Sname[2] = "CS002"
SId_to_Sname[3] = "CS003"
SId_to_Sname[4] = "CS004"
SId_to_Sname[5] = "CS005"
SId_to_Sname[6] = "CS006"
SId_to_Sname[7] = "CS007"
# SId_to_Sname[8] = "CS008"
# SId_to_Sname[9] = "CS009"
# SId_to_Sname[10] = "CS010"
SId_to_Sname[11] = "CS011"
# SId_to_Sname[12] = "CS012"
SId_to_Sname[13] = "CS013"
# SId_to_Sname[14] = "CS014"
# SId_to_Sname[15] = "CS015"
# SId_to_Sname[16] = "CS016"
SId_to_Sname[17] = "CS017"
# SId_to_Sname[18] = "CS018"
# SId_to_Sname[19] = "CS019"
# SId_to_Sname[20] = "CS020"
SId_to_Sname[21] = "CS021"
# SId_to_Sname[22] = "CS022"
# SId_to_Sname[23] = "CS023"
SId_to_Sname[24] = "CS024"
# SId_to_Sname[25] = "CS025"
SId_to_Sname[26] = "CS026"
# SId_to_Sname[27] = "CS027"
SId_to_Sname[28] = "CS028"
# SId_to_Sname[29] = "CS029"
SId_to_Sname[30] = "CS030"
SId_to_Sname[31] = "CS031"
SId_to_Sname[32] = "CS032"
SId_to_Sname[101] = "CS101"
# SId_to_Sname[102] = "CS102"
SId_to_Sname[103] = "CS103"
SId_to_Sname[121] = "CS201"
SId_to_Sname[141] = "CS301"
SId_to_Sname[142] = "CS302"
SId_to_Sname[161] = "CS401"
SId_to_Sname[181] = "CS501"

# SId_to_Sname[104] = "RS104"
# SId_to_Sname[105] = "RS105"
SId_to_Sname[106] = "RS106"
# SId_to_Sname[107] = "RS107"
# SId_to_Sname[108] = "RS108"
# SId_to_Sname[109] = "RS109"
# SId_to_Sname[122] = "RS202"
# SId_to_Sname[123] = "RS203"
# SId_to_Sname[124] = "RS204"
SId_to_Sname[125] = "RS205"
# SId_to_Sname[126] = "RS206"
# SId_to_Sname[127] = "RS207"
SId_to_Sname[128] = "RS208"
# SId_to_Sname[129] = "RS209"
SId_to_Sname[130] = "RS210"
# SId_to_Sname[143] = "RS303"
# SId_to_Sname[144] = "RS304"
SId_to_Sname[145] = "RS305"
SId_to_Sname[146] = "RS306"
SId_to_Sname[147] = "RS307"
# SId_to_Sname[148] = "RS308"
# SId_to_Sname[149] = "RS309"
SId_to_Sname[150] = "RS310"
SId_to_Sname[166] = "RS406"
SId_to_Sname[167] = "RS407"
SId_to_Sname[169] = "RS409"
SId_to_Sname[183] = "RS503"
SId_to_Sname[188] = "RS508"
SId_to_Sname[189] = "RS509"

SId_to_Sname[201] = "DE601"
SId_to_Sname[202] = "DE602"
SId_to_Sname[203] = "DE603"
SId_to_Sname[204] = "DE604"
SId_to_Sname[205] = "DE605"
SId_to_Sname[206] = "FR606"
SId_to_Sname[207] = "SE607"
SId_to_Sname[208] = "UK608"

# this just "inverts" the previous list, discarding unused values
Sname_to_SId_dict = {
    name: ID for ID, name in enumerate(SId_to_Sname) if name is not None
}


def even_antName_to_odd(even_ant_name):
    even_num = int(even_ant_name)
    odd_num = even_num + 1
    return str(odd_num).zfill(9)


def antName_is_even(ant_name):
    return not int(ant_name) % 2