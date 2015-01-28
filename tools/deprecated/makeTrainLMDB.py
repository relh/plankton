import lmdb

env = lmdb.open('../data/dbs')
train_db = env.open_db('train_db')
#test_db = env.open_db('test_db')

#with env.begin(db=train_db, write=True) as txn:
#	txn.put('key', 'value')


