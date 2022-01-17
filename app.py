from flask import Flask,request,flash, request, redirect, url_for, Response, jsonify
import pymongo
import json
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
import time

app = Flask(__name__)

client = pymongo.MongoClient("mongodb+srv://ISEdemo:databasepassword@cluster0.tpcqf.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db_name = 'Project_db'

prev_time = time.time()

def binary_search(index, a, pos, step):
  length = len(a)
  mid = int(length/2)
  first_list = [a[x] for x in range(mid)]
  second_list = [a[x] for x in range(mid,length)]
  step += 1
  if length == 1:
    return pos
  else:
    if index > first_list[len(first_list)-1]:
      return binary_search(index,second_list,pos+len(first_list),step)
    else:
      return binary_search(index,first_list,pos,step)

def get_percentile(x):
  data = list(client[db_name]['Dataset predictions'].find({},{'_id':False}))
  cleaned = []
  for i in data:
    cleaned.append(i['value'])
  return round((binary_search(x,sorted(cleaned),0,0)/len(cleaned))*100,1)

@app.route('/query_data')
def get_data():
  ret=dict()
  id=int(request.args.get('id'))
  try:
    filter_param=request.args.get('filp')
    filter_op=request.args.get('filo')
    real_op=''
    if (filter_op=='greater'):
      real_op='$gt'
    elif (filter_op=='lesser'):
      real_op='$lt'
    else:
      real_op='$eq'
    filter_val=int(request.args.get('filv'))
    try:
      sort_param=request.args.get('sortp')
      sort_by=request.args.get('sortb')
      if sort_by=='ASCENDING':
        data = client[db_name]['Main dataset'].find({filter_param:{real_op:filter_val}},{'_id':False}).sort(sort_param,pymongo.ASCENDING)
      else:
        data = client[db_name]['Main dataset'].find({filter_param:{real_op:filter_val}},{'_id':False}).sort(sort_param,pymongo.DESCENDING)
    except:
      data = client[db_name]['Main dataset'].find({filter_param:{real_op:filter_val}},{'_id':False})
  except:
    try:
      sort_param=request.args.get('sortp')
      sort_by=request.args.get('sortb')
      if sort_by=='ASCENDING':
        data = client[db_name]['Main dataset'].find({},{'_id':False}).sort(sort_param,pymongo.ASCENDING)
      else:
        data = client[db_name]['Main dataset'].find({},{'_id':False}).sort(sort_param,pymongo.DESCENDING)
    except:
      data = client[db_name]['Main dataset'].find({},{'_id':False})
  ret['data']=list(data)[id:id+20]
  print(len(list(data)))
  return jsonify(ret)

@app.route('/get_corr')
def get_corr():
  ret=dict()
  params = []
  for i in range(6):
    params.append(request.args.get('p'+str(i)))
  df = pd.read_csv('diabetes1.csv')
  df = df.corr().reset_index().drop('index',axis=1)
  df.columns = [str(i) for i in range(22)]
  ret['data']=[
               round(df[params[0]][int(params[1])],3),
               round(df[params[2]][int(params[3])],3),
               round(df[params[4]][int(params[5])],3),
  ]
  return jsonify(ret)

@app.route('/get_history')
def get_history():
  ret=dict()
  db=client[db_name]
  temp_list = list(db['User prediction'].find({},{'_id': False}))
  length=len(temp_list)
  if length-10 >= 0:
    ret['data']=temp_list[len(temp_list)-10:len(temp_list)]
  else:
    ret['data']=temp_list
  return jsonify(ret)

@app.route('/get_prediction')
def get_prediction():
  global prev_time
  params = []
  for i in range(1,22):
    params.append(request.args.get('q'+str(i)))

  clf = joblib.load("model.pkl")
  p = np.array([params])
  the_data = []
  the_data.append(clf.predict(p)[0])
  the_data.append(list(clf.staged_predict_proba(p))[-1:][0][0][0])
  the_data.append(list(clf.staged_predict_proba(p))[-1:][0][0][1])
  the_data.append(get_percentile(list(clf.staged_predict_proba(p))[-1:][0][0][1]))
  ret={'data':the_data}
  if (time.time()-prev_time > 5):
    user_dict={}
    for i in range(22): #rebuild list to history
      if i==0:
        user_dict[chr(ord('a')+i)]=clf.predict(p)[0]
        user_dict[chr(ord('a')+i+1)]=list(clf.staged_predict_proba(p))[-1:][0][0][0]
      else:
        user_dict[chr(ord('a')+i+1)]=int(params[i-1])
    db=client[db_name]
    db['User prediction'].insert_one(user_dict)
    prev_time=time.time()
    return jsonify(ret)
  else:
    return jsonify(ret)