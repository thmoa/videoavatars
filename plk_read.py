import pickle as pkl

plk_file = 'camera.pkl'

with open(plk_file, 'rb') as fp:
    u = pkl._Unpickler(fp)
    u.encoding = 'latin1'
    data = u.load()
    print(data)
