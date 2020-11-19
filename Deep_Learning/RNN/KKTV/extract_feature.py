import pandas as pd
import numpy as np
import time
import datetime
import copy
import math
import csv

ep_diff = pd.read_csv('diff_ep_information.csv')

for k in range(45) :
    print('starting data-0{:0>2d}.csv'.format(k+1))
    df = pd.read_csv('data-0{:0>2d}.csv'.format(k+1))
    single_feature = []
    for i in range(34):
        single_feature.append([])
        for j in range(38):
            single_feature[i].append(0)
    #print(single_feature)

    people_indlist = []

    ystart = datetime.datetime(2017, 1, 1)

    now_id = -1
    v_count = 0
    count = -1
    features = []
    for event in range(len(df)) :
        #catch which day
        stoptime = datetime.datetime.strptime(df['event_time'][event][:-3], '%Y-%m-%d %H:%M:%S.')
        past_Day = (stoptime - ystart)
        week = past_Day.days // 7 
        day = (past_Day.days % 7 + 6) % 7#sunday monday ...

        if df['user_id'][event] != now_id :
            now_id = df['user_id'][event]
            people_indlist.append(now_id)
            count+=1
            features.append(copy.deepcopy(single_feature))
            v_count = 0
            now_video = df['title_id'][event]
            now_ep = df['episode_number'][event]
            temp_video = [[now_video], [[now_ep]]]

        #counting how many record this week
        features[count][week][37] += 1

        dstart = datetime.datetime(year = stoptime.year, month = stoptime.month, day = stoptime.day)
        passtime = (stoptime - dstart).total_seconds()
        playdur = df['played_duration'][event]
        timeslot = day * 4

        if df['title_id'][event] != now_video :
            now_video = df['title_id'][event]
            now_ep = df['episode_number'][event]
            v_count += 1
            temp_video[0].append(now_video)
            temp_video[1].append([now_ep])
        else :
            if df['episode_number'][event] != now_ep :
                now_ep = df['episode_number'][event]
                temp_video[1][v_count].append(now_ep)

        #print(temp_video)
        if now_ep > df['series_total_episodes_count'][event] :
            features[count][week][34] = 0.0
        else :
            features[count][week][34] = min(1, max(features[count][week][34], len(temp_video[1][v_count]) / (6 + math.sqrt(df['series_total_episodes_count'][event]))))
        
        if week >= 0 and week <= 32 and ep_diff[str(df['title_id'][event])][week] >= 1 :
            if now_ep == df['series_total_episodes_count'][event] :
                features[count][week][36] += 1
            elif now_ep < df['series_total_episodes_count'][event] :
                features[count][week][36] += (now_ep / df['series_total_episodes_count'][event])
        '''
        print('passtime : ', passtime, sep = '')
        print('week : ', week, sep = '')
        print('day : ', day, sep = '')
        print('timeslot : ', timeslot, sep = '')
        print('playdur : ', playdur, sep = '')
        '''
        if df['internet_connection_type'][event][:8] == 'cellular' : # 1, 0 for cellular; 0, 1 for wifi.
            features[count][week][32] += 1
            #features[count][week][33] = 0
        elif df['internet_connection_type'][event] == 'wifi' :
            #features[count][week][32] = 0
            features[count][week][33] += 1
        #else :
        #    features[count][week][32] = 0
        #    features[count][week][33] = 0

        if df['platform'][event] == 'iOS' :
            features[count][week][29] += 1
            #features[count][week][30] = 0
            #features[count][week][31] = 0
        elif df['platform'][event] == 'Android' :
            #features[count][week][29] = 0
            features[count][week][30] += 1
            #features[count][week][31] = 0
        elif df['platform'][event] == 'Web' :
            #features[count][week][29] = 0
            #features[count][week][30] = 0
            features[count][week][31] += 1


        if passtime >= 3600 and passtime < 32400 :
            #print('slot1')
            if passtime - playdur < 3600 :
                if timeslot == 0 :
                    features[count][week-1][27] += 3600 - passtime + playdur
                else :
                    features[count][week][timeslot - 1] += 3600 - passtime + playdur
                features[count][week][timeslot] += passtime - 3600
            else :
                features[count][week][timeslot] += playdur

        elif passtime >= 32400 and passtime < 61200 :
            #print('slot2')
            if passtime - playdur < 32400 :
                features[count][week][timeslot] += 32400 - passtime + playdur
                features[count][week][timeslot + 1] += passtime - 32400
            else :
                features[count][week][timeslot + 1] += playdur
        elif passtime >= 61200 and passtime < 75600 :
            #print('slot3')
            if passtime - playdur < 61200 :
                features[count][week][timeslot + 1] += 61200 - passtime + playdur
                features[count][week][timeslot + 2] += passtime - 61200
                if features[count][week][timeslot + 1] < 0 :
                    print('----ErrorA----')
                if features[count][week][timeslot + 2] < 0 :
                    print('----ErrorB----')
            else :
                features[count][week][timeslot + 2] += playdur
        elif passtime >= 75600 and passtime < 86400:
            #print('slot4')
            if passtime - playdur < 75600 :
                features[count][week][timeslot + 2] += 75600 - passtime + playdur
                features[count][week][timeslot + 3] += passtime - 75600
                if features[count][week][timeslot + 2] < 0 :
                    print('----ErrorA----')
                if features[count][week][timeslot + 3] < 0 :
                    print('----ErrorB----')
            else :
                features[count][week][timeslot + 3] += playdur
        elif passtime >= 0 and passtime < 3600 :
            if passtime - playdur < 0 :
                if timeslot == 0 :
                    if passtime - playdur < -10800 :
                        features[count][week - 1][26] += playdur - passtime - 10800
                        features[count][week - 1][27] += 10800 + passtime
                    else :
                        features[count][week - 1][27] += playdur
                else :
                    if passtime - playdur < -10800:
                        features[count][week][timeslot - 2] += playdur - passtime - 10800
                        features[count][week][timeslot - 1] += passtime + 10800
                    else :
                        features[count][week][timeslot - 1] += playdur
            else :
                if timeslot == 0 :
                    features[count][week - 1][27] += playdur
                else :
                    features[count][week][timeslot - 1] += playdur
    top3 = [[16, 76, 38], [16, 76, 58], [16, 76, 47], [76, 47, 16], [124, 76, 47], [47, 76, 105], [47, 105, 76], [47, 76, 105], [47, 45, 105], [45, 47, 130], [45, 47, 105], [45, 47, 105], [45, 105, 47], [45, 47, 105], [45, 142, 47], [45, 14, 69], [14, 45, 69], [14, 69, 45], [14, 69, 45], [14, 73, 69], [14, 73, 69], [14, 73, 53], [73, 53, 77], [73, 53, 77], [73, 53, 77], [73, 53, 77], [53, 73, 74], [53, 73, 74], [74, 77, 73], [74, 77, 79], [74, 77, 79], [74, 79, 14], [74, 79, 148], [148, 51, 74]]
    now_id = -1
    v_count = 0
    count = -1
    for event in range(len(df)) :
        #catch which day
        stoptime = datetime.datetime.strptime(df['event_time'][event][:-3], '%Y-%m-%d %H:%M:%S.')
        past_Day = (stoptime - ystart)
        week = past_Day.days // 7 
        day = (past_Day.days % 7 + 6) % 7#sunday monday ...

        if df['user_id'][event] != now_id :
            now_id = df['user_id'][event]
            count+=1
            v_count = 0
            now_video = df['title_id'][event]
            now_ep = df['episode_number'][event]
        playdur = df['played_duration'][event]

        if df['title_id'][event] != now_video :
            now_video = df['title_id'][event]
            now_ep = df['episode_number'][event]
            v_count += 1
        else :
            if df['episode_number'][event] != now_ep :
                now_ep = df['episode_number'][event]

        if df['title_id'][event] in top3[week] :
            features[count][week][35] += playdur

        if features[count][week][29] > features[count][week][30] and features[count][week][29] > features[count][week][31] :
            features[count][week][29] = 1
            features[count][week][30] = 0
            features[count][week][31] = 0
        elif features[count][week][30] > features[count][week][29] and features[count][week][30] > features[count][week][31] :
            features[count][week][29] = 0
            features[count][week][30] = 1
            features[count][week][31] = 0
        elif features[count][week][31] > features[count][week][30] and features[count][week][31] > features[count][week][29] :
            features[count][week][29] = 0
            features[count][week][30] = 0
            features[count][week][31] = 1
        else :
            features[count][week][29] = 0
            features[count][week][30] = 0
            features[count][week][31] = 0
        if features[count][week][32] > features[count][week][33] :
            features[count][week][32] = 1
            features[count][week][33] = 0
        elif features[count][week][33] > features[count][week][32] :
            features[count][week][32] = 0
            features[count][week][33] = 1
        else :
            features[count][week][32] = 0
            features[count][week][33] = 0
    split_week = np.reshape(features, (len(people_indlist) * 34, 38))
    for i in range(len(split_week)) :
        for j in range(28) :
            temp = split_week[i][j] ** 0.2 / (28800 ** 0.2)
            split_week[i][j] = max(0, min(1, temp))
        temp = split_week[i][35] ** 0.2 / (28800 ** 0.2)
        split_week[i][35] = max(0, min(1, temp))
        if split_week[i][37] != 0 :
            split_week[i][36] /= split_week[i][37]
            if i % 34 != 33:
                split_week[i + 1][28] = 1
    pindex = []

    for i in range(len(people_indlist)) :
        for j in range(34) :
            pindex.append(people_indlist[i])

    ndf = pd.DataFrame(split_week[:,:-1], columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36'],index = pindex)
    
    if k == 0 :
        ndf.to_csv('feature.csv', columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36'])
    else :
        ndf.to_csv('feature.csv', mode='a', header=False)