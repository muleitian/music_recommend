
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix


#基于人气的推荐
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {user_id: 'score'},inplace=True)
    
        
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    
    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations
    

#基于物品相似度的推荐
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given user
    #获取给指定用户唯一的条目（歌曲）
    def get_user_items(self, user):
        # 得到数据中给指定用户听过的所有歌
        user_data = self.train_data[self.train_data[self.user_id] == user]
        # 得到上述歌曲的列表
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Get unique users for a given item (song)
    #找出物品（歌）的所有用户的信息
    def get_item_users(self, item):
        # 找出对应歌的所有数据
        item_data = self.train_data[self.train_data[self.item_id] == item]
        # set() 函数创建一个无序不重复元素集,找出对应歌曲的所有用户
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #Get unique items (songs) in the training data
    #在训练数据集中产生所有的物品（歌曲）
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    #Construct cooccurence matrix
    #构建共存的矩阵
    def construct_cooccurence_matrix(self, user_songs, all_songs):
            
        
        # 所有用户的集合，其中每一个列表对应的一首歌的所有用户
        user_songs_users = []
        # 遍历用户听过的所有歌曲，找出每个歌曲对应的所有的用户
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
            
        
        # 构造行为用户通过的歌，列为所有的歌的矩阵
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        
        for i in range(0,len(all_songs)):
            #Calculate unique listeners (users) of song (item) i
            # 找出第i首歌的数据
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            # 找出听过第i首歌的听过的用户
            users_i = set(songs_i_data[self.user_id].unique())
            # 对用户听过的歌遍历
            for j in range(0,len(user_songs)):       
                    
                
                #每个j对应的每首歌的所有用户
                users_j = user_songs_users[j]
                    
                
                # intersection() 方法用于返回两个或更多集合中都包含的元素，即交集。
                users_intersection = users_i.intersection(users_j)
                
                
                if len(users_intersection) != 0:
                    
                    # union() 方法返回两个集合的并集
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    
    
    #产生推荐
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        # 稀疏矩阵中不为0的值
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        # axis=1表示最终元素个数与行数相同, axis=0表示最终个数与列数相同
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        # user_sim_scores 是一个列表，每个元素是按列相加的值/推荐用户中听的所有歌曲的数量
        #tolist（）将矩阵（matrix）和数组（array）转化为列表。
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        # (e,i)对应的（值，索引）
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        #产生10个物品的推荐
        rank = 1 
        for i in range(0,len(sort_index)):
            if np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    
    def recommend(self, user):
        
        
        user_songs = self.get_user_items(user)    
        # 输出给待推荐用户听过歌曲的数量    
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        
        # 输出数据集中所有的歌曲
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        
        # 构造稀疏矩阵
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    
    #通过物品推荐得到相似的物品，传入参数为物品的列表
    def get_similar_items(self, item_list):
        user_songs = item_list
        
        
        #得到训练样本的所有歌曲
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations

#基于SVD分解
class SVD:
    def __init__(self):
        self.train_data = None
        self.coo_matrix = None
        self.K = 50
        self.uTest = None
    def create(self,train_data,uTest):
        self.train_data = train_data
        self.uTest = uTest
        song_merged = self.train_data[['user', 'listen_count']].groupby('user').sum().reset_index()
        song_merged.rename(columns={'listen_count': 'total_listen_count'}, inplace=True)
        song = pd.merge(train_data, song_merged)
        song['fractional_play_count'] = song['listen_count'] / song['total_listen_count']
        small_set = song
        user_codes = small_set.user.drop_duplicates().reset_index()
        song_codes = small_set.song.drop_duplicates().reset_index()
        user_codes.rename(columns={'index': 'user_index'}, inplace=True)
        song_codes.rename(columns={'index': 'song_index'}, inplace=True)
        song_codes['so_index_value'] = list(song_codes.index)
        user_codes['us_index_value'] = list(user_codes.index)
        small_set = pd.merge(small_set, song_codes, how='left')
        small_set = pd.merge(small_set, user_codes, how='left')
        mat_candidate = small_set[['us_index_value', 'so_index_value', 'fractional_play_count']]
        data_array = mat_candidate.fractional_play_count.values
        row_array = mat_candidate.us_index_value.values
        col_array = mat_candidate.so_index_value.values
        self.coo_matrix = coo_matrix((data_array, (row_array, col_array)), dtype=float)
        self.train_data = small_set
    def compute_svd(self):
        urm = self.coo_matrix
        U, s, Vt = svds(urm, self.K)
        dim = (len(s), len(s))
        S = np.zeros(dim, dtype=np.float32)
        for i in range(0, len(s)):
            S[i,i] = mt.sqrt(s[i])

        U = csc_matrix(U, dtype=np.float32)
        S = csc_matrix(S, dtype=np.float32)
        Vt = csc_matrix(Vt, dtype=np.float32)

        return U, S, Vt

    def compute_estimated_matrix(self):
        U,S,Vt = self.compute_svd()
        urm = self.coo_matrix
        uTest = self.uTest
        MAX_PID = urm.shape[1]
        MAX_UID = urm.shape[0]
        rightTerm = S*Vt
        max_recommendation = 250
        estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
        recomendRatings = np.zeros(shape=(MAX_UID,max_recommendation ), dtype=np.float16)
        for userTest in uTest:
            prod = U[userTest, :]*rightTerm
            estimatedRatings[userTest, :] = prod.todense()
            recomendRatings[userTest, :] = (estimatedRatings[userTest, :]).argsort()[:max_recommendation] 
        return recomendRatings

class user_popularity:
        def __init__(self):
            self.train_data = None
            self.user_Id = None
            self.data = {}

        def create(self, train_data, user_Id):
            self.train_data = train_data
            self.user_Id = user_Id

        def Eucliden(self, user1, user2):
            user1_data = self.data[user1]
            user2_data = self.data[user2]
            distance = 0
            for key in user1_data.keys():
                if key in user2_data.keys():
                    distance += np.power((user1_data[key] - user2_data[key]), 2)
            return 1 / (1 + np.sqrt(distance))

        def top10_simliar(self, userId):
            res = []
            for userid in self.data.keys():
                if userid != userId:
                    similar = self.Eucliden(userId, userid)
                    res.append((userid, similar))
            res.sort(key=lambda val: val[1])
            res1 = res[:2]
            user1 = res1[0][0]
            user2 = res1[1][0]
            sm_r1 = list(self.data[user1])
            sm_r2 = list(self.data[user2])
            sm_r1.extend(sm_r2)
            d1 = []
            for i in sm_r1:
                if i not in self.data[userId].keys():
                    d1.append(i)
            return d1[:3]

        def recommend(self):
            for i in range(len(self.data)):
                if not self.train_data.loc[i]['user'] in self.data.keys():
                    self.data[self.train_data.loc[i]['user']] = {
                        self.train_data.loc[i]['title']: self.train_data.loc[i]['listen_count']}
                else:
                    self.data[self.train_data.loc[i]['user']][self.train_data.loc[i]['title']] = self.train_data.loc[i][
                        'listen_count']
            return self.top10_simliar(self.user_Id)
