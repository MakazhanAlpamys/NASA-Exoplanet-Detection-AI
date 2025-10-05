from data_preprocessing import ExoplanetDataProcessor
p = ExoplanetDataProcessor()
toi = p.select_features(p.load_toi_dataset(), 'toi')
k2  = p.select_features(p.load_k2_dataset(), 'k2')
print('koi_duration in TOI:', 'koi_duration' in toi.columns)
print('koi_duration in K2:',  'koi_duration' in k2.columns)
