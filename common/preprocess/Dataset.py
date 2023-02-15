import math
import torch
import numpy as np
from   tick.base import TimeFunction
from   tick.hawkes import HawkesKernelExp, SimuHawkesExpKernels, SimuInhomogeneousPoisson
import pickle
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def cluster(X, eps=0.3, min_samples=10, if_plot=True, cluster_type=None):
    X = StandardScaler().fit_transform(X)
    # plt.scatter(X[:,0],X[:,1],s=1,marker='*', color='green')
    # plt.show()
    # exit()
    # Compute DBSCAN
    eps=0.15
    min_samples=8
    if cluster_type=='dbscan':
        db = DBSCAN(eps=eps, min_samples=min_samples ).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # exit()
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_centers = np.zeros((n_clusters_,2))
        for i in range(n_clusters_):
            cluster_centers[i] = np.average(X[labels==i], axis=0)
        # plt.scatter(cluster_centers[:,0], cluster_centers[:,1])
        # plt.show()
        # exit()
        # count=0
        for i in range(labels.shape[0]):
            if labels[i]==-1:
                # count+=1
                labels[i] = np.argmin(np.linalg.norm(cluster_centers-X[i], axis=1))
                # exit()
                # print(i,np.min(np.linalg.norm(cluster_centers-X[i])))
                #  
        # print(count)


        # print(n_clusters_)
        # exit()
        n_noise_ = list(labels).count(-1)

        if if_plot:
            for i in range(n_clusters_):
                points = X[labels==i]
                plt.scatter(points[:,0], points[:,1], label='cluster'+str(i),s=3)
            plt.legend()
            plt.savefig('Particle_dbscan_eps_0_15_min_samples_8.pdf')
            plt.close()
            # plt.show()
        # exit()
        print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)
        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        # print(
        #     "Adjusted Mutual Information: %0.3f"
        #     % metrics.adjusted_mutual_info_score(labels_true, labels)
        # )
        # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
        return  labels, n_clusters_
    if cluster_type=='agglomerative':
        clustering = AgglomerativeClustering().fit(X)
        return []

        # >>> clustering
        # AgglomerativeClustering()
        # >>> clustering.labels_
        # array([1, 1, 1, 0, 0, 0])


class MakeData():

    def __init__(self, num_changepoints, data_mode=None, num_sequences=1, run_time=500,  underlying="hawkes", generation='uniform', dist_cp = 0):
        self.num_changepoints = num_changepoints
        # print('y')
        self.num_sequences = num_sequences
        if data_mode == 'real':
            # print('y')
            return
        self.run_time = run_time
        self.underlying = underlying
        self.data_mode = data_mode
        if generation=='uniform':
            self.chp_range = [ [x, x+(2/5)*(1/(num_changepoints+1))]  for x in np.linspace(0,1, num_changepoints+2)[1:-1] ]
            self.baselines = [ 1, 2 ] * int((num_changepoints+1)/2) 
            if len(self.baselines) < num_changepoints+1:
                self.baselines += [1]
            self.adjacencies = np.random.uniform(.4,.7,num_changepoints+1)
        # if generation=='vary_dist_of_coeff':
        if generation=='vary_dist_btw_cp':
            self.chp_range = [ [x, x+0.0001] for x in [.3, .3 + dist_cp] ]
            # print('chp_range',self.chp_range)
            # return 
            # exit()
            self.baselines = [ 1, 2 ] * int((num_changepoints+1)/2) 
            if len(self.baselines) < num_changepoints+1:
                self.baselines += [1]
            self.adjacencies = np.random.uniform(.4,.7,num_changepoints+1)
            
        if generation=='vary_diff_dist':
            self.chp_range = [ [x, x+0.01] for x in [.4] ]
            # print('chp_range',self.chp_range)
            # return 
            # exit()
            self.baselines = []
            for i in range(5):
                for j in range(20):
                    self.baselines += [[1.0,1.0 + .2*(i+1)]]
            # self.baselines = 
            # for x in self.baselines:
            #     print(x)
            # exit()
            self.adjacencies = [np.random.uniform(.4,.7,2) for _ in range(num_sequences)]
        
        if generation=='vary_diff_dist_coef':
            # print('y')
            self.chp_range = [ [x, x+0.01] for x in [.4] ]
            # print('chp_range',self.chp_range)
            # return 
            # exit()
            self.baselines, self.adjacencies = [],[]
            for i in range(5):
                for j in range(20):
                    self.baselines += [[1.0,1.2]]
                    self.adjacencies += [[.4,.4+(i+1)*.1]]
            # for x in self.adjacencies:
                # print(x)
            # print('y')
            # exit()
        if generation=='vary_dim':
            self.chp_range = [ [x, x+0.01] for x in [.4] ]
            self.baselines, self.adjacencies = [],[]
            for i in range(5):
                for j in range(20):
                    self.baselines += [ [ [1.0+j*0.1 for j in range(i+1) ],[1.5+j*0.1 for j in range(i+1) ] ]]
                    if j==0:
                        adj1 = np.random.uniform(.08,.1,(i+1,i+1))
                        adj2 = np.random.uniform(.06,.1,(i+1,i+1))                        
                    self.adjacencies += [ [ adj1, adj2 ] ]
        if generation=='vary_len':
            self.chp_range = [ [x, x+0.01] for x in [.3,.7] ]
            self.baselines, self.adjacencies, self.run_time = [],[],[]
            for i in range(6):
                for j in range(30):
                    self.baselines += [[1.0,1.5,2]]
                    self.adjacencies += [np.random.uniform(.4,.7,num_changepoints+1)]
                    self.run_time += [200*(i+1)]
            # exit()un_time)
            # continu
            # for x in self.run_time:
            #     print(x)
            # exit()
            # for x in self.adjacencies:# baselines:
            #     print(x[0])
            #     print(x[1])
            #     print('*'*10)
            # exit()

    def render_diff_dist(self):
        self.data_dict = {"datas":[], "intensities":[], "intensity_times":[], "change_points":[], "run_times":[]}
        n_nodes = 1
        for i in  range(self.num_sequences):
            data = torch.Tensor([])
            ints = np.array([])
            ints_t = np.array([])
            chps = [0]
            if isinstance(self.run_time,list):
                run_time = self.run_time[i]
            else:
                run_time = self.run_time
            # print(run_time)
            # continue
            for (low,high) in self.chp_range:
                rn = low + (high - low)*np.random.random()
                chp = run_time*rn
                chps.append(chp)
            chps.append(run_time)
            for j,chp in enumerate(chps[:-1]):
                if isinstance(self.baselines[i][j], list):
                    baseline = self.baselines[i][j]
                    n_nodes = len(baseline)
                else:
                    baseline = [self.baselines[i][j]]

                if type(self.adjacencies[i][j]) is np.ndarray: #isinstance(self.adjacencies[i][j], )np.ndarray:
                    adjacency = self.adjacencies[i][j]# baseline = self.baselines[i][j]
                else:
                    adjacency = np.ones((n_nodes,n_nodes))*self.adjacencies[i][j]

                # print(baseline)
                # print(adjacency)
                # continue
                # exit()                
                decay = np.ones((n_nodes, n_nodes))
                process = SimuHawkesExpKernels(baseline=baseline, adjacency=adjacency, decays=decay, end_time=chps[j+1]-chp, verbose = False)
                process.track_intensity(0.1)
                process.simulate()
                intensity = process.tracked_intensity
                intensity_time = process.intensity_tracked_times + chp
                data = torch.cat((data, torch.Tensor(process.timestamps[0] + chp)))
                ints = np.hstack([ints,intensity[0]])
                ints_t = np.hstack([ints_t, intensity_time])
            # continue
            self.data_dict["datas"].append(data)
            print(data.shape)
            self.data_dict["intensities"].append(ints)
            self.data_dict["intensity_times"].append(ints_t)
            self.data_dict["change_points"].append(chps[1:-1])
            self.data_dict["run_times"].append(run_time)
        # exit()
        # for x in self.data_dict['datas']:
        #     print(x.size())
        # print(self.data_dict['datas'])

        return self.data_dict
    
    def render(self):
        self.data_dict = {"datas":[], "intensities":[], "intensity_times":[], "change_points":[], "run_times":[]}
        n_nodes = 1
        for i in range(self.num_sequences):
            data = torch.Tensor([])
            ints = np.array([])
            ints_t = np.array([])
            chps = [0]
            for i,(low,high) in enumerate(self.chp_range):
                rn = low + (high - low)*np.random.random()
                chp = self.run_time*rn
                chps.append(chp)
            chps.append(self.run_time)
            for i,chp in enumerate(chps[:-1]):
                baseline = [self.baselines[i]]
                if self.underlying == "hawkes":
                    adjacency = np.ones((n_nodes,n_nodes))*self.adjacencies[i]
                    decay = np.ones((n_nodes, n_nodes))
                    process = SimuHawkesExpKernels(baseline=baseline, adjacency=adjacency, decays=decay, end_time=chps[i+1]-chp, verbose = False)
                if self.underlying == "poisson":
                    T = np.arange(0,1.5*(chps[i+1]-chp),(chps[i+1]-chp)/20)
                    Y = np.ones(T.shape[0])*baseline
                    tf = TimeFunction((T,Y), dt = 0.01)
                    process = SimuInhomogeneousPoisson([tf], end_time=chps[i+1]-chp, verbose=False)
                process.track_intensity(0.1)
                process.simulate()
                intensity = process.tracked_intensity
                intensity_time = process.intensity_tracked_times + chp
                data = torch.cat((data, torch.Tensor(process.timestamps[0] + chp)))
                ints = np.hstack([ints,intensity[0]])
                ints_t = np.hstack([ints_t, intensity_time])
            self.data_dict["datas"].append(data)
            self.data_dict["intensities"].append(ints)
            self.data_dict["intensity_times"].append(ints_t)
            self.data_dict["change_points"].append(chps[1:-1])
            self.data_dict["run_times"].append(self.run_time)
        for x in self.data_dict['datas']:
            print(x.size())
        # print(self.data_dict['datas'])

        return self.data_dict

    def show(self, fig=False):
        datas = self.data_dict["datas"]
        change_points = self.data_dict["change_points"]
        intensities = self.data_dict["intensities"]
        intensity_times = self.data_dict["intensity_times"]
        print("Number of sequences: ", self.num_sequences)
        print("Runtime of the sequence: ", self.run_time)
        for i in range(self.num_sequences):
            print("   Number of elements in the sequence {}: ".format(i), len(datas[i]))
            print("   Change_points in the sequence {}: ".format(i),[round(ch,2) for ch in change_points[i]])

        if fig:
            plt.figure()
            plt.plot(intensity_times[0], intensities[0])
            plt.xlabel("Runtime of the sequence")
            plt.ylabel("Intensity")
            plt.savefig("Sequence_cp"+str(self.num_changepoints)+'.pdf')
            plt.show()
        
    def save(self, file):    
        outfile = open(file, 'wb')
        pickle.dump(self.data_dict, outfile)
        outfile.close()
        
    def load(self, file):
        infile = open(file, 'rb')
        self.data_dict = pickle.load(infile)
        infile.close()
        return self.data_dict

class ProcessRealData():

    def __init__(self,data_file=None):
        self.data_file= data_file

    def get_particle_data(self, file, run_time=100):
        df=pd.read_excel(file)
        easting=4
        northing=4
        cp = [221, 242, 282]
        df['e_cell'] = (easting * (df['Easting'] - df['Easting'].min() + 1)/(df['Easting'].max() - df['Easting'].min() + 1) ).apply(math.ceil)
        df['n_cell'] = (northing * (df['Northing'] - df['Northing'].min() + 1)/(df['Northing'].max() - df['Northing'].min() + 1) ).apply(math.ceil)
        df['cell']=northing*(df['e_cell']-1)+df['n_cell']
        df['e_cell_loc'] = df['Easting'].min() + (df['e_cell']-0.5)*(df['Easting'].max() - df['Easting'].min())/easting
        df['n_cell_loc'] = df['Northing'].min() + (df['n_cell']-0.5)*(df['Northing'].max() - df['Northing'].min())/northing
        for i in range(1,easting+1):
            for j in range(1,northing+1): 
                e_loc = df['Easting'].min() + (i-0.5)*(df['Easting'].max() - df['Easting'].min())/easting
                n_loc = df['Northing'].min() + (j-0.5)*(df['Northing'].max() - df['Northing'].min())/northing
                df['len_'+str(northing*(i-1)+j)] = ((df['Easting']-e_loc)**2 + (df['Northing']-n_loc)**2)**0.5
        
        times= df['Time'].values
        times = (times/times[-1])*run_time
        intensity = np.zeros(df['Time'].shape[0])
        num_samples = 10
        for i in range(len(times)):
            num_s = min(min(len(times)-i-1, num_samples), min(i, num_samples))
            intensity[i] = 2*num_s / (times[i+num_s] - times[i-num_s] + 1e-10)
        
        time_scaler = MinMaxScaler((0,run_time))
        feature_scaler = StandardScaler()
        data_dict = {}
        #data_dict['time'] = [df['Time'].values]
        data_dict['time'] = torch.Tensor(time_scaler.fit_transform(df[['Time']].values)).T
        # data_dict['datas'] = [torch.Tensor(times)]
        # data_dict['datas_mark'] = [torch.Tensor(df['cell'].values).to(torch.int64) - 1] 
        # data_dict['time'] = torch.Tensor([g + i * 0.01 for k, group in groupby(df['Time']) for i, g in enumerate(group)]).unsqueeze(dim=1)
        data_dict['mark'] = [torch.Tensor(df['cell'].values).to(torch.int64)]
        data_dict["intensities"] = [intensity]
        data_dict["intensity_times"] = [times]
        data_dict["change_points"] = [times[cp].tolist()]
        features = torch.Tensor(df.drop(['Time', 'e_cell','n_cell','e_cell_loc','n_cell_loc','cell'], axis = 1).values)
        data_dict['features'] = [torch.Tensor(feature_scaler.fit_transform(features))]
        data_dict['num_types'] = [easting * northing]
        data_dict['len_seq'] = [data_dict['time'].shape[1]]
        data_dict['len_feat'] = [features.shape[1]]
        data_dict["run_times"] = [run_time]
        return data_dict

    
    def get_intensity(self, t, delta):
        min_t,max_t=torch.min(t), torch.max(t)
        interval, count, lower, upper=delta,0,min_t, min_t+delta
        list_of_count, list_of_upper=[], []
        for x in t:
            if x > lower and x <= upper:
                count+=1
            if x > upper:
                list_of_count += [count]
                list_of_upper += [upper]
                count, lower, upper = 0, lower+interval, upper+interval
        return list_of_count, list_of_upper

    def get_rat_data(self, data_dict_init):
        data = data_dict_init['data']
        cp = data_dict_init['cp']
        data_dict = {}
        data_dict['time'] = [torch.from_numpy(seq['data']) for seq in data]
        data_dict['mark'] = [torch.ones(seq['data'].shape[0]).type(torch.int64) for seq in data] 
        data_dict['features'] = [torch.ones((seq['data'].shape[0],1)) for seq in data] 
        data_dict["intensity_times"] = [seq['intensity_times'] for seq in data]
        data_dict["intensities"] = [seq['intensity'] for seq in data]
        data_dict["change_points"] = [cp for seq in data]
        data_dict['num_types'] = [1 for seq in data]
        data_dict['len_seq'] = [seq['data'].shape[0] for seq in data]
        data_dict['len_feat'] = [1 for seq in data]
        data_dict["run_times"] = [seq['data'][-1] for seq in data]
        print(data_dict['len_seq'])
        return data_dict

    def get_earthquake_data(self, dict, run_time = 100, bins=400, Latitude=7, Longitude=7, clip=False):
        # exit()
        df = pd.DataFrame(dict)
        if clip:
            df = pd.DataFrame(df.loc[7000:]).reset_index().drop(['index'],axis=1)
            cp = [50.2]
        else:
            df = pd.DataFrame(df.loc[6000:]).reset_index().drop(['index'],axis=1)
            cp = [57.9]
        # print(df['Latitude'].min(), df['Latitude'].max())

        df['e_cell'] = (Latitude * (df['Latitude'] - df['Latitude'].min() + 1)/(df['Latitude'].max() - df['Latitude'].min() + 1) ).apply(lambda x:math.ceil(round(x,6)))
        df['n_cell'] = (Longitude * (df['Longitude'] - df['Longitude'].min() + 1)/(df['Longitude'].max() - df['Longitude'].min() + 1) ).apply(lambda x:math.ceil(round(x,6)))
        
        # x = df['e_cell'].values
        # y=df['n_cell'].values
        # plt.scatter(x,y)
        # plt.show()
        # print('Max', x.max(), ' Min: ', x.min())
        # exit()

        df['cell']=Longitude*(df['e_cell']-1)+df['n_cell']
        df['e_cell_loc'] = df['Latitude'].min() + (df['e_cell']-0.5)*(df['Latitude'].max() - df['Latitude'].min())/Latitude
        df['n_cell_loc'] = df['Longitude'].min() + (df['n_cell']-0.5)*(df['Longitude'].max() - df['Longitude'].min())/Longitude
        for i in range(1,Latitude+1):
            for j in range(1,Longitude+1): 
                e_loc = df['Latitude'].min() + (i-0.5)*(df['Latitude'].max() - df['Latitude'].min())/Latitude
                n_loc = df['Longitude'].min() + (j-0.5)*(df['Longitude'].max() - df['Longitude'].min())/Longitude
                df['len_'+str(Longitude*(i-1)+j)] = ((df['Latitude']-e_loc)**2 + (df['Longitude']-n_loc)**2)**0.5
        
        times= torch.Tensor(df['Time'].values)
        times_norm = ((times-times[0])/(times[-1]-times[0]))*run_time
        list_of_count, list_of_upper = self.get_intensity(times,bins)
        list_of_upper = torch.Tensor(list_of_upper)

        # print(type(list_of_count), type(list_of_upper))
        # plt.plot(list_of_upper, list_of_count)
        # plt.savefig('EQ_clipped_intensity.pdf')
        # print(min(list_of_upper), max(list_of_upper))
        # plt.show()
        # exit()


        list_of_upper = ((list_of_upper-times[0])/(times[-1]-times[0]))*run_time
        time_scaler = MinMaxScaler((0,run_time))
        feature_scaler = StandardScaler()
        
        data_dict = {}
        data_dict['time'] = torch.Tensor(time_scaler.fit_transform(df[['Time']].values)).T
        # data_dict['datas'] = [torch.Tensor(times_norm)]
        # data_dict['datas_mark'] = [torch.Tensor(df['cell'].values).to(torch.int64) - 1]
        # data_dict['time'] = torch.Tensor([g + i * 0.01 for k, group in groupby(df['Time']) for i, g in enumerate(group)]).unsqueeze(dim=1)
        data_dict['mark'] = [torch.Tensor(df['cell'].values).to(torch.int64)]
        data_dict["intensities"] = [np.array(list_of_count)]
        data_dict["intensity_times"] = [np.array(list_of_upper)]
        # plt.plot(list_of_upper, list_of_count)
        # plt.savefig('EQ_clipped_intensity.pdf')
        # plt.show()
        data_dict["change_points"] = [cp]
        features = torch.Tensor(df.drop(['Time', 'e_cell','n_cell','e_cell_loc','n_cell_loc','cell'], axis = 1).values)
        data_dict['features'] = [torch.Tensor(feature_scaler.fit_transform(features))]
        data_dict['num_types'] = [Latitude * Longitude]
        data_dict['len_seq'] = [data_dict['time'].shape[1]]
        data_dict['len_feat'] = [features.shape[1]]
        data_dict["run_times"] = [run_time]
        return data_dict

    def get_earthquake_data_dbscan_cluster(self, dict, run_time = 100, bins=400, Latitude=7, Longitude=7, clip=False):
        
        df = pd.DataFrame(dict)
        if clip:
            df = pd.DataFrame(df.loc[7000:]).reset_index().drop(['index'],axis=1)
            cp = [50.2]
        else:
            df = pd.DataFrame(df.loc[6000:]).reset_index().drop(['index'],axis=1)
            cp = [57.9]
        coordinates_of_earthquake = np.array([df['Latitude'].values, df['Longitude'].values]).T
        # print(coordinates_of_earthquake.shape)
        earthquake_center_labels, n_clusters = cluster( coordinates_of_earthquake, eps=0.1, min_samples=25, if_plot=True, cluster_type='dbscan')
        earthquake_centers = np.zeros((n_clusters,2))
        for i in range(n_clusters):
            earthquake_centers[i] = np.average(coordinates_of_earthquake[earthquake_center_labels==i], axis=0)
        
        # print(earthquake_center_labels.shape)
        # print(earthquake_center_labels.min())
        # print(earthquake_center_labels.max())
        # plt.plot(earthquake_center_labels)
        # plt.show()
        # exit()

        # df['e_cell'] = (Latitude * (df['Latitude'] - df['Latitude'].min() + 1)/(df['Latitude'].max() - df['Latitude'].min() + 1) ).apply(lambda x:math.ceil(round(x,6)))
        # df['n_cell'] = (Longitude * (df['Longitude'] - df['Longitude'].min() + 1)/(df['Longitude'].max() - df['Longitude'].min() + 1) ).apply(lambda x:math.ceil(round(x,6)))
        
        # x = df['e_cell'].values
        # y=df['n_cell'].values
        # plt.scatter(x,y)
        # plt.show()
        # print('Max', x.max(), ' Min: ', x.min())
        # exit()

        df_cell=earthquake_center_labels
        # df['e_cell_loc'] = df['Latitude'].min()# + (df['e_cell']-0.5)*(df['Latitude'].max() - df['Latitude'].min())/Latitude
        # df['n_cell_loc'] = df['Longitude'].min() + (df['n_cell']-0.5)*(df['Longitude'].max() - df['Longitude'].min())/Longitude
        for i,cc in enumerate(earthquake_centers):
            df['len_'+str(i)] = ((df['Latitude']-cc[0])**2 + (df['Longitude']-cc[1])**2)**0.5
        # exit()
        times= torch.Tensor(df['Time'].values)
        times_norm = ((times-times[0])/(times[-1]-times[0]))*run_time
        list_of_count, list_of_upper = self.get_intensity(times,bins)
        list_of_upper = torch.Tensor(list_of_upper)
        list_of_upper = ((list_of_upper-times[0])/(times[-1]-times[0]))*run_time
        time_scaler = MinMaxScaler((0,run_time))
        feature_scaler = StandardScaler()
        
        data_dict = {}
        data_dict['time'] = torch.Tensor(time_scaler.fit_transform(df[['Time']].values)).T
        # data_dict['datas'] = [torch.Tensor(times_norm)]
        # data_dict['datas_mark'] = [torch.Tensor(df['cell'].values).to(torch.int64) - 1]
        # data_dict['time'] = torch.Tensor([g + i * 0.01 for k, group in groupby(df['Time']) for i, g in enumerate(group)]).unsqueeze(dim=1)
        data_dict['mark'] = [torch.Tensor(df_cell).to(torch.int64)+1]
        data_dict["intensities"] = [np.array(list_of_count)]
        data_dict["intensity_times"] = [np.array(list_of_upper)]
        data_dict["change_points"] = [cp]
        features = torch.Tensor(df.drop(['Time'], axis = 1).values)
        data_dict['features'] = [torch.Tensor(feature_scaler.fit_transform(features))]
        data_dict['num_types'] = [n_clusters]
        data_dict['len_seq'] = [data_dict['time'].shape[1]]
        data_dict['len_feat'] = [features.shape[1]]
        data_dict["run_times"] = [run_time]
        return data_dict


    def get_adele_data(self,data_dict, run_time = 100):
        data_adele = data_dict['data']
        d = []
        for event in data_adele:
            d.append(event[2])      # 0 to max
        cp_adele = [a / run_time for a in list(set(data_dict['change_points']))] 
        col_names = ['Time','Mark']
        exp_dict = {}
        n_attr = max(d) - min(d) + 1
        for i in range(n_attr):
            col_names.append('feat'+str(i+1))
        for event in data_adele:
            if (event[0],event[1]) not in exp_dict :
                exp_dict[event[0],event[1]] = np.zeros(n_attr+2)
                exp_dict[event[0],event[1]][0] = event[0]
                exp_dict[event[0],event[1]][1] = event[1]
                exp_dict[event[0],event[1]][event[2]+2] = event[3]
            else:
                exp_dict[event[0],event[1]][event[2]+2] = event[3]
        df_adele = pd.DataFrame.from_dict(exp_dict, orient ='index').reset_index().drop(['index'],axis=1)
        df_adele.columns = col_names
        time_scaler = MinMaxScaler((0,run_time))
        feature_scaler = StandardScaler()
        real_dict = {}
        real_dict['time'] = torch.Tensor(time_scaler.fit_transform(df_adele[['Time']].values)).T
        # real_dict['datas'] = [torch.Tensor(df_adele['Time'].values)]
        # real_dict['datas_mark'] = [torch.Tensor(df_adele['Mark'].values).to(torch.int64)]
        real_dict['mark'] = [torch.Tensor(df_adele['Mark'].values).to(torch.int64) + 1] 
        real_dict["intensities"] = [data_dict['data_to_plot']['y-axis']]
        real_dict["intensity_times"] = [[x / run_time for x in data_dict['data_to_plot']['x-axis']]]
        real_dict["change_points"] = [cp_adele]
        features = torch.Tensor(df_adele.drop(['Time', 'Mark'], axis = 1).values)
        real_dict['features'] = [torch.Tensor(feature_scaler.fit_transform(features))]
        real_dict['num_types'] = [max(real_dict['mark'][0]+1)]
        real_dict['len_seq'] = [real_dict['time'].shape[1]]
        real_dict['len_feat'] = [features.shape[1]]
        real_dict["run_times"] = [run_time]
        return real_dict

    def preprocess_real_data(self,data_dict):
        self.data_dict = data_dict
        #self.data_dict["change_points"]=self.num_changepoints
        #self.data_dict = {"datas":[], "change_points":[], "run_times":[]}
        #data_excel = pd.read_excel(file)
        #data = torch.Tensor(data_excel['Time'])
        #data=torch.Tensor([g + i * 0.01 for k, group in groupby(data) for i, g in enumerate(group)])
        #self.data_dict["datas"].append(data)
        time_scaler = MinMaxScaler((0,100))
        # time_scaler = MinMaxScaler((0,self.data_dict['time'][-1].numpy())) # for adele data
        self.data_dict['time_norm'] = torch.Tensor(time_scaler.fit_transform(self.data_dict['time']))

        feature_scaler = StandardScaler().fit(self.data_dict['features'])
        self.data_dict['features_norm'] = torch.Tensor(feature_scaler.transform(self.data_dict['features']))

        self.data_dict['time_scaler'] = time_scaler
        self.data_dict['feature_scaler'] = feature_scaler
        self.data_dict["run_times"] = self.data_dict['time'][-1,0]
        return self.data_dict

    def preprocess_synthetic_data(self,data_dict, no_mark=None):
        # time_scaler = MinMaxScaler((0,run_time))
        feature_scaler = StandardScaler()
        num_seq = len(data_dict['datas'])
        data_dict['len_seq']  = [x.size()[0] for x in data_dict['datas']]
        data_dict['len_feat'] = [1] * num_seq 
        # print(list(data_dict['datas'][0].size() )[0], )
        # print( int(data_dict['datas'][0].shape[0]))
        data_dict['features'] = [torch.ones((x.shape[0],1)) for x in data_dict['datas']] 
        # for x in data_dict['features']:
        #     print(x.shape)
        # exit()
        
        data_dict['time'] = [x for x in data_dict['datas']]
        # torch.Tensor(time_scaler.fit_transform(x.numpy())) ]  
        if no_mark: # derive it from 'datas'
            data_dict['mark'] = [torch.ones(x.size()).type(torch.int64) for x in data_dict['datas']] 
            data_dict['num_types'] = [1] * num_seq        
        # print(data_dict['mark'][0][0].dtype)
        return data_dict

    def reprocess_data_ltr(self, data, reduce_label=False):
        # sequence = 0
        # ['time', 'mark', 'intensities', 'intensity_times', 'change_points', 'features', 'num_types', 'len_seq', 'len_feat', 'run_times'])
        n_seq = 5
        data['time'] = data['time'][:n_seq]
        data['mark'] = data['mark'][:n_seq]
        data['intensities'] = data['intensities'][:n_seq]
        data['intensity_times'] = data['intensity_times'][:n_seq]
        data['change_points'] = data['change_points'][:n_seq]
        data['features'] = data['features'][:n_seq]
        data['num_types'] = data['num_types'][:n_seq]
        data['len_seq'] = data['len_seq'][:n_seq]
        data['len_feat'] = data['features'][:n_seq]
        data['run_times'] = data['run_times'][:n_seq]

        # for m,t in zip(data['mark'][:5], data['time'][:5]):
        #     m = m[t>60]
        #     mark_labels = torch.unique(m, sorted=True)
        #     hist_mark_labels = [m[m==l].shape[0] for l in mark_labels]
        #     plt.plot(mark_labels, hist_mark_labels,label=str(i))
        #     plt.grid(True)
        #     i+=1
        # plt.legend()
        # plt.show()
        # exit()
        if reduce_label:
            label_cluster = torch.Tensor([\
            [1,3,5,8,17],\
            [1,6,7,8,17],\
            [1,5,7,8,17],\
            [1,3,5,8,17],\
            [1,9,10,11,17]])
            print(data['mark'][0][0].dtype)
            data['mark'] = [torch.Tensor([ torch.where(m_i < l)[0][0].item() for m_i in m ]).to(torch.int64) for m,l in zip(data['mark'], label_cluster)]
            print(data['mark'][0][0].dtype)

            # print(data['num_types'][0].dtype)
            data['num_types'] = [4 for _ in data['num_types']]
            # print(data['num_types'][0].dtype)

        data['mark'] = [m[t>60] for t,m in zip(data['time'], data['mark'])]
        print(data['mark'][0][0].dtype)

        data['intensities'] = [i[t>60] for i,t in zip(data['intensities'], data['intensity_times'])]
        data['intensity_times'] = [t[t>60]-60 for t in data['intensity_times']]
        
        data['change_points'] = [ [x-60 for x in cp] for cp in data['change_points']]
        # print(data['features'][0].shape)
        # exit()
        print(data['features'][0][0][0].dtype)
        data['features'] = [f[t>60] for t,f in zip(data['time'],data['features'])]
        print(data['features'][0][0][0].dtype)
        
        # print(data['features'][0].shape)
        # exit()
        
        data['time'] = [t[t>60]-60 for t in data['time']]
        # print(data['time'][0].shape)
        # exit()
        data['len_seq'] = [t.shape[0] for t in data['time']]
        data['len_feat'] = [f.shape[1] for f in data['features']]
        data['run_times'] = [t[-1] for t in data['time']]
        # exit()
        return data_dict

    def reprocess_data(self, data, reduce_label=False):
        # sequence = 0
        # ['time', 'mark', 'intensities', 'intensity_times', 'change_points', 'features', 'num_types', 'len_seq', 'len_feat', 'run_times'])

        t = data['time']
        cp = data['change_points']
        print(len(t))
        
        for i, t_i in enumerate(t):
            plt.plot(t_i[:-1],t_i[1:] - t_i[:-1])
            plt.axvline(cp[i][0], color='red')
            plt.savefig('Arithmat'+str(i)+'.png')
            plt.close()
        exit()



        if reduce_label:
            label_cluster = torch.Tensor([1,5,9,13,17])
            print(data['mark'][0][0].dtype)
            data['mark'] = [torch.Tensor([ torch.where(m_i < label_cluster)[0][0].item() for m_i in m ]).to(torch.int64) for m in data['mark']]
            print(data['mark'][0][0].dtype)
            # exit()
            # print(data['num_types'][0].dtype)
            data['num_types'] = [4 for _ in data['num_types']]
            # print(data['num_types'][0].dtype)

        list_of_start_time=[]
        list_of_start_index=[]
        i=0
        for t,cp,run_time in zip(data['time'], data['change_points'], data['run_times']): 
            cp_i = cp[0]
            if i in [20, 35]:
                # list_of_start_time += [torch.Tensor(0).to(torch.int64)]
                # list_of_start_index += [torch.Tensor(0).to(torch.int64)]
                cp_i -= 40
           
            x = run_time - cp_i
            start_time = max(0,cp_i - x)
            # if cp_i < x:
            #     print('yes')
            list_of_start_time += [start_time]
            print(i, start_time, cp_i, run_time, t[0], t[-1])
            # print()
            list_of_start_index += [torch.where(t>start_time)[0][0].item()]
            i+=1
                
        # exit()
        data['mark'] = [m[i:] for m,i in zip(data['mark'], list_of_start_index)]
        # print(data['mark'][0][0].dtype)

        # print(data['intensity_times'][0][0].dtype)
        # print(list_of_start_time[0].dtype)
        # exit()
        data['intensities'] = [i[t>s.item()] for i,t,s in zip(data['intensities'], data['intensity_times'], list_of_start_time)]
        data['intensity_times'] = [t[t>s.item()]-s.item() for t,s in zip(data['intensity_times'], list_of_start_time)]
            

        data['change_points'] = [ [x-s for x in cp] for cp,s in zip(data['change_points'], list_of_start_time)]
        # print(data['features'][0].shape)
        # exit()
        # print(data['features'][0][0][0].dtype)
        data['features'] = [f[i:] for f,i in zip(data['features'], list_of_start_index)]
        # print(data['features'][0][0][0].dtype)
        
        # print(data['features'][0].shape)
        # exit()
        
        data['time'] = [ t[i:]-s for t,i,s in zip(data['time'], list_of_start_index, list_of_start_time)]
        # print(data['time'][0].shape)
        # exit()
        data['len_seq'] = [t.shape[0] for t in data['time']]
        data['len_feat'] = [f.shape[1] for f in data['features']]
        data['run_times'] = [t[-1] for t in data['time']]
        # exit()

        i=0
        for t in data['time']:
            print(i,t.shape)
            i+=1

        for cp, run_time, t in zip(data['change_points'], data['run_times'], data['time']):
            print(cp, run_time, t[-1]) 
        return data_dict

    def rescale_Anesth(self, data):
        
        # data['intensities'] = [i[t>s.item()] for i,t,s in zip(data['intensities'], data['intensity_times'], list_of_start_time)]
        # data['intensity_times'] = [t[t>s.item()]-s.item() for t,s in zip(data['intensity_times'], list_of_start_time)]
        i=0
        data['time'] = [t - t[0] for t, run_time in zip(data['time'], data['run_times'])]
        data['time'] = [t*(100.0/run_time) for t, run_time in zip(data['time'], data['run_times'])]
        data['change_points'] = [ cp*(100.0/run_time) for cp, run_time in zip(data['change_points'], data['run_times'])]
        data['run_times'] = [t[-1] for t in data['time']]

        s = [0,0,0,1,1,1,0,0]
        # print(len(data['change_points']))
        data['change_points'] = [torch.Tensor([cp[s[i]], cp[s[i]+2]])  for i,cp in enumerate(data['change_points'])]


        for x,t,cp in zip(data['run_times'], data['time'], data['change_points']):
            print(x,t[-1])
            # plt.plot(t)
            # plt.savefig('Time'+str(i)+'.png')
            for y in cp:
                plt.axvline(y,color='red')
        
            plt.plot(t[1:],t[1:]-t[:-1])
            plt.savefig('Interarrival'+str(i)+'.png')
            
            plt.close()
            i+=1
        # exit()

        
        for cp, run_time, t in zip(data['change_points'], data['run_times'], data['time']):
            print(cp, run_time, t[-1]) 
        return data_dict
    

    def rescale_Anesth_2cp(self, data):
        # for i in range(8):
        #     print(i,data['features'][i].shape, data['time'][i].shape)
        # exit()
        # data['intensities'] = [i[t>s.item()] for i,t,s in zip(data['intensities'], data['intensity_times'], list_of_start_time)]
        # data['intensity_times'] = [t[t>s.item()]-s.item() for t,s in zip(data['intensity_times'], list_of_start_time)]
        # i=0
        data['time'] = [t - t[0] for t, run_time in zip(data['time'], data['run_times'])]
        data['time'] = [t*(100.0/run_time) for t, run_time in zip(data['time'], data['run_times'])]
        data['change_points'] = [ cp*(100.0/run_time) for cp, run_time in zip(data['change_points'], data['run_times'])]
        data['run_times'] = [t[-1] for t in data['time']]
        # for cp in data['change_points']:
        #     print(cp)
        # exit()
        
        i=0
        # start_time = [0,0,42,42,0,0,42,0]
        # finish_time = [50,50,100,100,50,50,100,50]
        # s = [0,0,2,2,0,0,2,0]
        # ['time', 'mark', 'intensities', 'intensity_times', 'change_points', 'features', 'num_types', 'len_seq', 'len_feat', 'run_times'])
        for i in range(8):#
            if i in [0,1,4,5,7]:
                start_time=0
                finish_time = 57
                run_time = finish_time - start_time
                t = data['time'][i]
                t = t[t<finish_time]
                end_index = t.shape[0]
                t = t * (100.0/run_time)
                data['time'][i]=t
                # data['run_times'][i]=50                
                data['change_points'][i] = data['change_points'][i][:2] * (100.0/run_time)
                data['mark'][i] = data['mark'][i][:end_index]
                # print(data['features'][i].shape)
                # exit()
                # print(data['time'][i].shape)
                # print(data['mark'][i].shape)
                data['features'][i] = data['features'][i][:end_index]
                # print(data['features'][i].shape)
                # exit()
                # print(data['len_feat'][i])
                data['len_seq'][i] = t.shape[0]
                data['run_times'][i] = t[-1]
                # exit()
            else:
                t = data['time'][i]
                n=t.shape[0]
                start_time = 41
                finish_time = t[-1]
                run_time = finish_time - start_time
                t = t[t>start_time] - start_time
                start_index=n-t.shape[0]
                t = t * (100.0/run_time)
                data['time'][i]=t
                data['run_times'][i]=t[-1]                
                data['change_points'][i] = (data['change_points'][i][2:] - start_time) * (100.0/run_time)
                data['mark'][i] = data['mark'][i][start_index:]
                data['features'][i] = data['features'][i][start_index:]
                data['len_seq'][i] = t.shape[0]   
        i=0
        for x,t,cp in zip(data['run_times'], data['time'], data['change_points']):
            print(x,t[-1])
            for y in cp:
                plt.axvline(y,color='red')
            plt.plot(t[1:],t[1:]-t[:-1])
            plt.savefig('Interarrival'+str(i)+'.png')
            plt.close()
            i+=1
        # for i in range(8):
        #     print(i,data['features'][i].shape, data['time'][i].shape)
        # exit()
        
        return data_dict 

    def rescale_Anesth_adhoc(self, data):
        
        # for i in range(8):
        #     print(i,data['features'][i].shape, data['time'][i].shape)
        # exit()
        
        for i in range(8):#
            flag=False 
            # if i in [7]:
            #     start_time=11
            #     finish_time = 111
            #     flag=True 
            # if i in [5,7]:
            #     start_time=10
            #     finish_time = 100
            #     flag=True 

            if i==7:
                data['time'][i] = data['time'][i] - 11
                # run_time = finish_time - start_time
                # t = data['time'][i]
                # print(t.shape)
                # n=t.shape[0]
                # t = t[t<finish_time]
                # end_index = t.shape[0] 
                # t = t[t>start_time]
                # start_index = end_index - t.shape[0]
                # t = t * (100.0/run_time)
                # data['time'][i]=t
                # data['run_times'][i]=50                
                data['change_points'][i] = data['change_points'][i] - 11
                data['run_times'][i] = data['time'][i][-1]
                # data['mark'][i] = data['mark'][i][start_index:end_index]
                # print(i,data['features'][i].shape, n)
                # exit()
                # data['features'][i] = data['features'][i][start_index:end_index]
                # print(data['features'][i].shape)
                # print(data['time'][i].shape)
                # exit()
                
                # data['len_seq'][i] = t.shape[0]   
            # else:
                # print(i,data['features'][i].shape, data['time'][i].shape)
                
            
        # i=0
        # for x,t,cp,f in zip(data['run_times'], data['time'], data['change_points'], data['features']):
        #     # print(x,t[-1], t.shape[0])
        #     # plt.plot(t)
        #     # plt.savefig('Time'+str(i)+'.png')
        #     for y in cp:
        #         plt.axvline(y,color='red')
        
        #     plt.plot(t[1:],t[1:]-t[:-1])
        #     plt.savefig('Interarrival'+str(i)+'.png')
            
        #     plt.close()
        #     i+=1
            # print(t.shape, f.shape)
        # for i in range(8):
        #     print(i,data['features'][i].shape, data['time'][i].shape)
        # exit()
        return data_dict

        
    def get_particle_data_cluster(self, file, run_time=100):
        df=pd.read_excel(file)
        # for cl in df.columns:
        #     print(cl)
        # exit()
        coordinates= np.array([df['Northing'].values, df['Easting'].values]).T
        labels, n_clusters = cluster( coordinates, eps=0.15, min_samples=8, if_plot=True, cluster_type='dbscan')
        exit()
        centers = np.zeros((n_clusters,2))
        for i in range(n_clusters):
            centers[i] = np.average(coordinates[labels==i], axis=0)
        
        df_cell=labels
        for i,cc in enumerate(centers):
            df['len_'+str(i)] = ((df['Northing']-cc[0])**2 + (df['Easting']-cc[1])**2)**0.5
        
        cp = [221, 242, 282]

        times= df['Time'].values
        times = (times/times[-1])*run_time
        intensity = np.zeros(df['Time'].shape[0])
        num_samples = 10
        for i in range(len(times)):
            num_s = min(min(len(times)-i-1, num_samples), min(i, num_samples))
            intensity[i] = 2*num_s / (times[i+num_s] - times[i-num_s] + 1e-10)
        
        time_scaler = MinMaxScaler((0,run_time))
        feature_scaler = StandardScaler()
        data_dict = {}
        data_dict['time'] = torch.Tensor(time_scaler.fit_transform(df[['Time']].values)).T
        data_dict['mark'] = [torch.Tensor(df_cell).to(torch.int64)+1]
        print(torch.min(data_dict['mark'][0]))
        print(torch.max(data_dict['mark'][0]))
        # exit()
        data_dict["intensities"] = [intensity]
        data_dict["intensity_times"] = [times]
        data_dict["change_points"] = [times[cp].tolist()]
        features = torch.Tensor(df.drop(['Time'], axis = 1).values)
        data_dict['features'] = [torch.Tensor(feature_scaler.fit_transform(features))]
        data_dict['num_types'] = [n_clusters]
        data_dict['len_seq'] = [data_dict['time'].shape[1]]
        data_dict['len_feat'] = [features.shape[1]]
        data_dict["run_times"] = [run_time]

        #----------------------------------------------------------------------
        return data_dict

    def get_smaller_segments(self, data_dict, segment_lengths, sequences):
        
        # print(data_dict[])
        data_dict_new = {}
        data_dict_new['time']=[]
        data_dict_new['mark']=[]
        data_dict_new['intensities']=[]
        data_dict_new['intensity_times']=[]
        data_dict_new['change_points']=[]
        data_dict_new['features']=[]
        data_dict_new['num_types']=[]
        data_dict_new['len_seq']=[]
        data_dict_new['len_feat']=[]
        data_dict_new['run_times']=[]

        for i in sequences:
            cp = data_dict['change_points'][i]
            t = data_dict['time'][i]

            # print(t[0],t[-1])
            # exit()
            # print(t[0],t[-1])
            for cp_i in cp:
                for l in segment_lengths:
                    l_half = int(l/2)
                    start_time = cp_i - l_half
                    end_time = cp_i + l_half 
                    if end_time > t[-1]:
                        end_time = t[-2]
                    if start_time < t[0]:
                        start_time = t[1]

                    # print(type(t))
                    start_index = torch.where(t>start_time)[0][0].item()
                    # print(start_time)
                    # print(t[1990:2000])
                    # print(start_index)
                    end_index = torch.where(t>end_time)[0][0].item()
                    # print(end_index)
                    # exit()
                    data_dict_new['time'] += [t[start_index:end_index]]
                    data_dict_new['mark'] += [data_dict['mark'][i][start_index:end_index]]
                    # other_cps 
                    cp_list=[]#[cp_i]
                    # print(start_time, end_time)
                    # exit()
                    for cp_x in cp:
                        if True: # cp_x != cp_i:
                            # print('cp', cp_x)
                            if cp_x > start_time and cp_x < end_time:
                                # print(cp_x)
                                cp_list += [cp_x]
                    # exit()
                    data_dict_new['change_points'] += [cp_list]
                    # print(data_dict['features'][i].shape)
                    # z = data_dict['features'][i][start_index:end_index]
                    # print(z.shape)
                    data_dict_new['features'] += [data_dict['features'][i][start_index:end_index]]
                    data_dict_new['num_types'] += [data_dict['num_types'][i]]
                    # print(data_dict['len_seq'])
                    data_dict_new['len_seq'] += [end_index - start_index]
                    data_dict_new['len_feat'] += [data_dict['len_feat'][i]]
                    data_dict_new['run_times'] += [(t[end_index-1]-t[start_index]).item()]
                    data_dict_new['intensities'] += [data_dict['intensities'][i]]
                    data_dict_new['intensity_times'] += [data_dict['intensity_times'][i]]
            # exit()
            key = 'time'
            print(data_dict_new[key][0].shape)
            print(data_dict_new[key][0][0], data_dict_new[key][0][-1])
            # print(type(data_dict[key][0]), type(data_dict_new[key][0]) )
            # print
            # exit()
        return data_dict_new 

    def rescale_data(self, data_dict):

        num_seq=len(data_dict['time'])
        for i in range(num_seq):
            # print(data_dict['time'])
            t = data_dict['time'][i] 
            # print(t[0])
            # exit()
            start_time = t[0]
            # print(start_time)
            # print(t[0], t[-1])
            t = t - start_time
            scale = 100.0 / t[-1]
            # print(scale)
            t = t * scale
            # print(scale) 
            # print(t[0], t[-1])
            data_dict['time'][i] = t 
            # print(data_dict['change_points'][i])
            # exit()
            # tmp = [(cp_i - start_time) * scale for cp_i in data_dict['change_points'][i]] 
            data_dict['change_points'][i] = [(cp_i - start_time) * scale for cp_i in data_dict['change_points'][i]] 
            # print(data_dict['change_points'][i])
            # exit()
            data_dict['run_times'][i] = t[-1].item()-t[0].item()
        # exit()


        # print( len(data_dict['time']) )

        for x in data_dict['change_points']:
            # print(x.shape)
            print(x)
            # print(x[0], x[-1])
            # pass
        # print(data_dict['change_points'])
        return data_dict
    
    
if __name__ == "__main__":

    real, synthetic = True, False
    if real:
        filename = 'crime'
        details_dict={'Earthquake':{'src':'Earthquake_dbscan_cluster_16','Sequences':[0]},\
                        'Arithmat':{'src':'Arithmat_final','Sequences':[2]},\
                        'crime':{'src':'crime_scaled','Sequences':[0]},\
                        'Rat':{'src':'Rat_scaled','Sequences':[1,4,7,10]},\
                        'Anesth':{'src':'Anesth_scaled','Sequences':[0]},\
                        'Particle':{'src':'Particle_cluster','Sequences':[0]}}
        
        datapath = '../../data/'+details_dict[filename]['src']
        with open(datapath + '.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        data_out = ProcessRealData().get_smaller_segments(data_dict, [80], details_dict[filename]['Sequences'])
        # data_out = ProcessRealData().get_smaller_segments(data_dict, [10,15,20,25,30,35], details_dict[filename]['Sequences'])

        # exit()
        data_scaled = ProcessRealData().rescale_data(data_out)
        write_path = '../../data/'+filename+'_scaled_80'
        with open(write_path+'.pkl','wb') as f:
            pickle.dump( data_out, f,  protocol=4)

    if synthetic:
        vary_cp, vary_len, vary_next_dist, vary_dist_btw_cp, vary_diff_dist, vary_diff_dist_coef, vary_dim, vary_len_new = False, False, False, False, False, False, False, True
        if vary_cp:
            for i in range(1,6):
                data_param = {'num_cp':i, 'num_seq':1, 'run_time':200, 'underlying':"hawkes"}
                file_to_save = '../../data/' + data_param['underlying'] + '_seq_' + str(data_param['num_seq']) + '_cp_' + str(data_param['num_cp']) + '_runtime_' + str(data_param['run_time'])
                make_data = MakeData(num_changepoints=data_param['num_cp'] , num_sequences=data_param['num_seq'] , run_time=data_param['run_time'], underlying=data_param['underlying'] )
            
                data_dict = make_data.render()
                make_data.show(fig=True)
                # exit()
                data_dict  = ProcessRealData().preprocess_synthetic_data(data_dict, no_mark=True)
                with open(file_to_save+'.pkl', 'wb') as f:
                    pickle.dump(data_dict, f)
        if vary_len:

            for i in [100,200,500,700,1000,1500,2000,2500]:
                data_param = {'num_cp':2, 'num_seq':10, 'run_time':i, 'underlying':"hawkes"}
                file_to_save = '../../data/' + data_param['underlying'] + '_seq_' + str(data_param['num_seq']) + '_cp_' + str(data_param['num_cp']) + '_runtime_' + str(data_param['run_time'])
                make_data = MakeData(num_changepoints=data_param['num_cp'] , num_sequences=data_param['num_seq'] , run_time=data_param['run_time'], underlying=data_param['underlying'] )            
                data_dict = make_data.render()
                make_data.show(fig=True)
                # exit()
                data_dict  = ProcessRealData().preprocess_synthetic_data(data_dict, no_mark=True)
                with open(file_to_save+'.pkl', 'wb') as f:
                    pickle.dump(data_dict, f)

        if vary_next_dist:
            for i in [100]: #,200,500,700,1000,1500,2000,2500]:
                data_param = {'num_cp':1, 'num_seq':10, 'run_time':i, 'underlying':"hawkes"}
                file_to_save = '../../data/' + data_param['underlying'] + '_seq_' + str(data_param['num_seq']) + '_cp_' + str(data_param['num_cp']) + '_runtime_' + str(data_param['run_time'])
                make_data = MakeData(num_changepoints=data_param['num_cp'] , num_sequences=data_param['num_seq'] , run_time=data_param['run_time'], underlying=data_param['underlying'] )            
                data_dict = make_data.render()
                make_data.show(fig=True)
                data_dict  = ProcessRealData().preprocess_synthetic_data(data_dict, no_mark=True)
                with open(file_to_save+'.pkl', 'wb') as f:
                    pickle.dump(data_dict, f)
        
        if vary_dist_btw_cp:
            for  i in range(2,8):#np.linspace(.1,.6,6):
                dist_cp = i * 0.05
                dist_runtime = int(500*(dist_cp))
                data_param = {'num_cp':2, 'num_seq':10, 'run_time':500, 'underlying':"hawkes",'dist_cp':dist_cp}
                file_to_save = '../../data/' + data_param['underlying'] + '_seq_' + str(data_param['num_seq']) + '_cp_' + str(data_param['num_cp']) + '_runtime_' + str(data_param['run_time']) + '_dist_cp_'  + str(dist_runtime) 
                make_data = MakeData(num_changepoints=data_param['num_cp'] , num_sequences=data_param['num_seq'] , run_time=data_param['run_time'], underlying=data_param['underlying'], generation = 'vary_dist_btw_cp', dist_cp = data_param['dist_cp'])            
                data_dict = make_data.render()
                # make_data.show(fig=True)
                data_dict  = ProcessRealData().preprocess_synthetic_data(data_dict, no_mark=True)
                with open(file_to_save+'.pkl', 'wb') as f:
                    pickle.dump(data_dict, f)

        if vary_diff_dist:
            data_param = {'num_cp':1, 'num_seq':100, 'run_time':500, 'underlying':"hawkes"}
            file_to_save = '../../data/' + data_param['underlying'] + '_seq_' + str(data_param['num_seq']) + '_cp_' + str(data_param['num_cp']) + '_runtime_' + str(data_param['run_time']) + '_vary_diff_dist'
            print(file_to_save)
            make_data = MakeData(num_changepoints=data_param['num_cp'] , num_sequences=data_param['num_seq'] , run_time=data_param['run_time'], underlying=data_param['underlying'], generation = 'vary_diff_dist')            
            data_dict = make_data.render_diff_dist()
            # make_data.show(fig=True)
            data_dict  = ProcessRealData().preprocess_synthetic_data(data_dict, no_mark=True)
            with open(file_to_save+'.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
        
        if vary_diff_dist_coef:
            data_param = {'num_cp':1, 'num_seq':100, 'run_time':500, 'underlying':"hawkes"}
            file_to_save = '../../data/' + data_param['underlying'] + '_seq_' + str(data_param['num_seq']) + '_cp_' + str(data_param['num_cp']) + '_runtime_' + str(data_param['run_time']) + '_vary_diff_dist_coef'
            print(file_to_save)
            make_data = MakeData(num_changepoints=data_param['num_cp'] , num_sequences=data_param['num_seq'] , run_time=data_param['run_time'], underlying=data_param['underlying'], generation = 'vary_diff_dist_coef')            
            data_dict = make_data.render_diff_dist()
            # make_data.show(fig=True)
            data_dict  = ProcessRealData().preprocess_synthetic_data(data_dict, no_mark=True)
            with open(file_to_save+'.pkl', 'wb') as f:
                pickle.dump(data_dict, f)

        if vary_dim:
            data_param = {'num_cp':1, 'num_seq':100, 'run_time':500, 'underlying':"hawkes"}
            file_to_save = '../../data/' + data_param['underlying'] + '_seq_' + str(data_param['num_seq']) + '_cp_' + str(data_param['num_cp']) + '_runtime_' + str(data_param['run_time']) + '_vary_dim'
            print(file_to_save)
            make_data = MakeData(num_changepoints=data_param['num_cp'] , num_sequences=data_param['num_seq'] , run_time=data_param['run_time'], underlying=data_param['underlying'], generation = 'vary_dim')            
            data_dict = make_data.render_diff_dist()
            # make_data.show(fig=True)
            data_dict  = ProcessRealData().preprocess_synthetic_data(data_dict, no_mark=True)
            with open(file_to_save+'.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
        if vary_len_new:
            data_param = {'num_cp':2, 'num_seq':180, 'run_time':0, 'underlying':"hawkes"}
            file_to_save = '../../data/' + data_param['underlying'] + '_seq_' + str(data_param['num_seq']) + '_cp_' + str(data_param['num_cp']) + '_runtime_' + str(data_param['run_time']) + '_vary_len'
            print(file_to_save)
            make_data = MakeData(num_changepoints=data_param['num_cp'] , num_sequences=data_param['num_seq'] , run_time=data_param['run_time'], underlying=data_param['underlying'], generation = 'vary_len')            
            data_dict = make_data.render_diff_dist()
            # make_data.show(fig=True)
            data_dict  = ProcessRealData().preprocess_synthetic_data(data_dict, no_mark=True)
            with open(file_to_save+'.pkl', 'wb') as f:
                pickle.dump(data_dict, f)



    if False:#real:
        data = 'Particle'
        if data == 'Anesth':

            # datapath = '../../data/Anesth'
            # with open(datapath + '.pkl', 'rb') as f:
            #     data_dict = pickle.load(f)
            # data_out = ProcessRealData().rescale_Anesth_2cp(data_dict)
            # write_path = '../../data/Anesth_scaled_2cp'
            # with open(write_path+'.pkl','wb') as f:
            #     pickle.dump( data_out, f,  protocol=4)
            # exit()

            # datapath = '../../data/Anesth_scaled_2cp'
            # with open(datapath + '.pkl', 'rb') as f:
            #     data_dict = pickle.load(f)
            # data_out = ProcessRealData().rescale_Anesth_adhoc(data_dict)
            # write_path = '../../data/Anesth_scaled_3cp_final'
            # with open(write_path+'.pkl','wb') as f:
            #     pickle.dump( data_out, f,  protocol=4)
            # 2 4 5 6 7

            datapath = '../../data/Anesth_scaled_3cp_final'
            with open(datapath + '.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            data_out = ProcessRealData().rescale_Anesth_adhoc(data_dict)
            write_path = '../../data/Anesth_scaled_3cp_final_1'
            with open(write_path+'.pkl','wb') as f:
                pickle.dump( data_out, f,  protocol=4)

        if data == 'Arithmat':
            datapath = '../../data/Arithmat'
            with open(datapath + '.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            data_out = ProcessRealData().reprocess_data(data_dict, reduce_label=True)
            write_path = '../../data/Arithmat_tr_reduce_label'
            with open(write_path+'.pkl','wb') as f:
                pickle.dump( data_out, f,  protocol=4)#pickle.HIGHEST_PROTOCOL)
        if data == 'rat':
            datapath = '../../data/rat_data_src'
            with open(datapath + '.pkl', 'rb') as f:
                data_dict = pickle.load(f)#, encoding='latin-1'
            data_out = ProcessRealData().get_rat_data(data_dict)
            write_path = '../../data/Rat'
            with open(write_path+'.pkl','wb') as f:
                pickle.dump( data_out, f,  protocol=4)#pickle.HIGHEST_PROTOCOL)
        if data == 'Particle':
            datapath = '../../data/particle-data.xlsx'
            process_data = ProcessRealData()
            data_out = process_data.get_particle_data_cluster(datapath)
            write_path = '../../data/Particle_cluster'
            with open(write_path+'.pkl','wb') as f:
                pickle.dump( data_out, f,  protocol=4)#pickle.HIGHEST_PROTOCOL)


        if data=='earthquake':
            datapath = '../../data/Earthquake_Italy_3Plus'
            with open(datapath + '.pkl', 'rb') as f:
                data_dict = pickle.load(f)#, encoding='latin-1'
            process_data = ProcessRealData(datapath)
            grid_clustering, dbscan_clustering, clip, clip_dbscan_clustering = False,False,False, False
            
            clip_dbscan_clustering = True 

            if grid_clustering:
                for i,j in zip([2,2,3,3,4],[2,3,2,4,3]):
                    data_out = process_data.get_earthquake_data(data_dict, Latitude=i, Longitude=j)
                    write_path = '../../data/Earthquake_' + str(i) + '_' + str(j)
                    with open(write_path+'.pkl','wb') as f:
                        pickle.dump( data_out, f,  protocol=4)#pickle.HIGHEST_PROTOCOL)
            if dbscan_clustering:
                data_out = process_data.get_earthquake_data_dbscan_cluster(data_dict)
                write_path = '../../data/Earthquake_dbscan_cluster_16'
                with open(write_path+'.pkl','wb') as f:
                    pickle.dump( data_out, f,  protocol=4)#pickle.HIGHEST_PROTOCOL)

            if clip:
                data_out = process_data.get_earthquake_data(data_dict, clip=True)
                write_path = '../../data/Earthquake_clipped'
                with open(write_path+'.pkl','wb') as f:
                    pickle.dump( data_out, f,  protocol=4)#pickle.HIGHEST_PROTOCOL)

            if clip_dbscan_clustering:
                data_out = process_data.get_earthquake_data_dbscan_cluster(data_dict, clip=True)
                write_path = '../../data/Earthquake_dbscan_cluster_clipped'
                with open(write_path+'.pkl','wb') as f:
                    pickle.dump( data_out, f,  protocol=4)#pickle.HIGHEST_PROTOCOL)
