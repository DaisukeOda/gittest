#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:42:05 2020

@author: 164404
"""

import joblib
import os , re

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# ---------------------------------------------------------

if (len(sys.argv[0]) == 0):
	print(r"develop")
	dev_mode = 1
	import os ; init_py = os.path.join(  r'C:\#Work\python_source\$include\init.py'  )
	if ( os.path.exists(init_py) ) : import codecs ; exec(codecs.open(  init_py   ,'r' ,'UTF-8').read())
	import os ; common_py = os.path.join( r'C:\Work\眠気推定_2019年度_簡易DS\common.py' )
	if ( os.path.exists(common_py) ) : import codecs ; exec(codecs.open(  common_py   ,'r' ,'UTF-8').read())
else:
	print("実行モード")
	dev_mode = 0


# ---------------------------------------------------------
# 設定ファイル
if (dev_mode == 0 ):
	# こっちは開発モード
	print(r"開発モード")
	home_dir = r"C:\DRMS"
	config_file = os.path.join(   r"C:\Work\眠気推定_2019年度_簡易DS\DRMS_config.ini"   )
else:
	# こっちは実行モード
	args = sys.argv
	home_dir =  args[1]
#	home_dir = r"C:\Temp" #debug
	# 設定ファイル名、ディレクトリ名は固定
	config_file = os.path.join(home_dir , r"DRMS_script" , r"DRMS_config.ini")

# ---------------------------------------------------------
# 設定ファイルをロードする
if ( os.path.exists(config_file) == False) :
	err_msg = config_file + r" が存在しません。終了します。"
	print( err_msg )
	sys.exit( err_msg )
config = pd.read_csv( config_file , sep='\t' , engine='python' , encoding='UTF8' , header=None )#, nrows=1000 )#,skiprows=10 )
# 設定ファイルをそのまま実行して、変数として定義する
for i in np.arange(0 , config.shape[0]) :
#	print(config.iloc[i][0])
	exec( config.iloc[i][0] )
#	print(  config.iloc[i][0] )

# common.py をロードする
if (dev_mode == 0 ):
	common_py = os.path.join(home_dir , common_py)
	if ( os.path.exists(common_py) ) :
		import codecs
		print(common_py + " をロード")
		exec(codecs.open(  common_py   ,'r' ,'UTF-8').read())
	else:
		err_msg = common_py + r" が存在しません。終了します。"
		print( err_msg )
		sys.exit( err_msg )
# ---------------------------------------------------------

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# =============================================================================
"""
model_dir=r"C:\Temp\DRMS_mode_files"
x_vars = os.path.join(model_dir , "x_vars.txt")
model_files=['randomforest_検証用被験者id_(0, 1).joblib'
,'randomforest_検証用被験者id_(0, 2).joblib'
,'randomforest_検証用被験者id_(0, 3).joblib'
,'randomforest_検証用被験者id_(0, 4).joblib'
,'randomforest_検証用被験者id_(1, 2).joblib'
,'randomforest_検証用被験者id_(1, 3).joblib'
,'randomforest_検証用被験者id_(1, 4).joblib'
,'randomforest_検証用被験者id_(2, 3).joblib'
,'randomforest_検証用被験者id_(2, 4).joblib'
,'randomforest_検証用被験者id_(3, 4).joblib'
]
#model_files = list( map (lambda x :  os.path.join(  model_dir , x  ) , model_files ) )
"""

def main(home_dir , input_file , save_dir , threshold ):
	try :
		
		########################################
		"""
		home_dir=r"C:\DRMS"
		input_file = r"C:\Work\眠気推定_2019年度_多治見テストコース実車実験\data\dataset_with_diff.txt.bz2"
		input_file = r"C:\Temp\DRMS_estimate_data\dataset_for_estimate_1577067212.485.txt.bz2"
		input_file = r"C:\Temp\DRMS_estimate_data\dataset_for_estimate_*.txt.bz2"
		input_file = r"C:\DRMS\DRMS_estimate_data\dataset_for_estimate_*.txt.bz2"
		threshold=2
		threshold=3
		source_path = r"C:\Work\眠気推定_2019年度_簡易DS"
	#	input_param = os.path.join( source_path , r'data' , 'クロス集計_眠気レベル' + str(threshold) +  '以上のパラメーター.txt.bz2' )
	#	input_file = r"C:\Temp\DRMS_estimate_data\dataset_for_estimate_*.txt*.bz2"
		model_dir= home_dir
		save_dir =  r"C:\DRMS"
		"""
		########################################
	#	model_dir= os.path.join(home_dir , r"DRMS_mode_files")
		model_dir= home_dir
		
		if (threshold == 2): model_dir= os.path.join(model_dir , model_files_2)
		if (threshold == 3): model_dir= os.path.join(model_dir , model_files_3)
		
		model_files = glob(  os.path.join(model_dir , "*.joblib"))
		input_param = os.path.join( model_dir , "クロス集計_眠気レベル"+ str(threshold) + "以上のパラメーター.txt.bz2" )
		
		os.chdir(home_dir)
		
		data_dir = home_dir
		
		# ------------------------------------------------------------
		#フルパスで渡された場合、一度分解してパスを組み立て直す
		input_file = os.path.join(data_dir , estimate_data , os.path.split(input_file)[1])
		input_files = np.sort(glob( input_file))
		if (  len(input_files) == 0 ) :
			err_msg = input_file + " が存在しません。終了します。"
			print(err_msg )
			sys.exit(err_msg)
		elif (  len(input_files) == 1 ) :
			input_file = input_files[-1] # 最新のファイルのみを取り出す
		elif (  len(input_files) >= 2 ) :
			err_msg = input_file + " が二つ以上存在します。1ファイルのみ対象です。ワイルドカードの設定を見直してください。終了します。"
			print(err_msg )
			sys.exit(err_msg)
		
		file_suffix = re.sub( r".*_([\d\.]+.*)" , r"\1" ,  os.path.splitext( os.path.splitext( os.path.split(input_file)[1] )[0])[0] )
		# ------------------------------------------------------------
		# 出力ファイルの保存場所の初期値はホームディレクトリ
		save_dir = os.path.join(save_dir , estimate_data)
		os.makedirs(save_dir  , exist_ok=True) #python3.2以降
		if not os.path.exists(save_dir) :
			err_msg = save_dir + r" が存在しないため、結果をファイルに保存できません。"
			print(err_msg )
			sys.exit(err_msg)
		save_file =  str(os.path.join (save_dir , r"result_drowsiness_for_more_than" +str(threshold)+ "_" + file_suffix + ".txt"))
		# ------------------------------------------------------------
		
		#---------------------------------------------------
		# 説明変数
		x_var_file = os.path.join ( model_dir , r"x_vars_org.txt")
		tempdf = pd.read_csv( x_var_file , sep='\t' , engine='python' , encoding='UTF8')#, nrows=1000 )#,skiprows=10 )
		#x_vars_of_fdt_eye = list(tempdf[tempdf.columns[0]])
		x_vars_of_fdt_eye = tempdf.copy()
		#---------------------------------------------------
		
		#####################################################################################
		
		tempdf = pd.read_csv( input_param , sep='\t' , engine='python' , encoding='UTF8')#, nrows=1000 )#,skiprows=10 )
		tempdf = tempdf.rename(columns={'x_col':'x_var'})
		x_var_cpt = tempdf.copy()
		# bin から上限・加減を取り出す
		x_var_cpt['lower_threshold'] = x_var_cpt['bin'].apply(  lambda x : float(re.sub(r"\[(.*),.*" , r"\1" , x)) )
		x_var_cpt = x_var_cpt.reset_index()
		temp = x_var_cpt.copy()
		temp = temp [['index', 'x_var','lower_threshold']]
		temp = temp.rename(columns = { 'lower_threshold':'upper_threshold'})
		temp['index'] = temp['index'] -1
		temp = pd.merge(x_var_cpt , temp , on = ['index' , 'x_var'] ,  how='outer' )
		temp = temp[  np.isnan(temp['bin'].str.len()) == False ].reset_index(drop=True)
		x_var_cpt = temp.copy()
		#####################################################################################
		
		
		#---------------------------------------------------
	#	input_file = os.path.join( r'C:\DRMS\DRMS_estimate_data\dataset_for_estimate_1577067212.485.txt.bz2' )
		dat = pd.read_csv( input_file , sep='\t' , engine='python' , encoding='UTF8')#, nrows=1000 )#,skiprows=10 )
		dat['variable'] = dat['variable'].apply( lambda x : re.sub(r"\.0分" , r"分" , x))
		
		# 説明変数をmergeすることで絞り込む
		dat =pd.merge(dat , x_vars_of_fdt_eye , left_on=['variable'] , right_on=list(x_vars_of_fdt_eye.columns) , how="right")
		
		key_cols = ['subject', 'unixtime', 'timezone', 'dir_nm']
		dat = dat.pivot_table(index= key_cols , columns='variable', values='value').reset_index(drop=False) # ここで(drop=True) キー列が消えてしまう
		
		# ----------------------------------
		# --閾値表から条件を取り出して、スコアをセットしていく
		#	set_col = 'high'
		#set_col = 'high_by_linear'
		set_col = 'high_by_quadratic'
		x_cols = list( x_vars_of_fdt_eye[ x_vars_of_fdt_eye.columns[0]] )
		#for x_col in x_vars_of_fdt_eye :
		for x_col in x_cols :
		#		x_col = x_cols[10] #debug
		#		print(x_col)
			cpt_temp = x_var_cpt[ x_var_cpt['x_var'] == x_col].reset_index(drop=True)
			new_val = dat[x_col] #* -1000 #debugように -1000を乗じた
			# --閾値表から条件を取り出して、スコアをセットしていく
			for i_cpt_temp in np.arange( 0 , cpt_temp.shape[0]):
		#			i_cpt_temp = 4
				lower_threshold = cpt_temp.iloc[i_cpt_temp ][ "lower_threshold"]
				upper_threshold = cpt_temp.iloc[i_cpt_temp][ "upper_threshold"]
				if (  np.isnan( upper_threshold)  == True):
		#				print(i_cpt_temp)
					new_val = np.where ( (lower_threshold  <= dat[x_col]) , cpt_temp.iloc[i_cpt_temp ][set_col]  , new_val )
				else:
					new_val = np.where ( (lower_threshold  <= dat[x_col])&(dat[x_col] <= upper_threshold) , cpt_temp.iloc[i_cpt_temp ][set_col]  , new_val )
			dat[x_col] = new_val
		# ----------------------------------
		# ot( dat.describe().T , index=True)
		new_dat = dat.copy()
		dat_evacuation = dat.copy()
		x_cols_evacuation = x_vars_of_fdt_eye
		
		#####################################################################################
		
		new_dat = dat_evacuation.copy()
		min_value = 0.75 #最良　（眠気レベル2以上の判別の場合、3以上の場合も）
		
		for x_col in  x_cols:
		#	new_dat[x_col] = np.where( new_dat[x_col] >= min_value  , new_dat[x_col]  , 0) #元のまま
			new_dat[x_col] = np.where( new_dat[x_col] >= min_value  , 1  , 0) # F値は小数点第3位レベルでこっちのほうがいい
		# ↑ , 眠気レベル2以上の判別の場合、スコアの閾値は0.7であれば、1に置換しても、元の値のままでも判別モデルの性能はほぼ変わらない
		# ↑ , 眠気レベル3以上の判別の場合、スコアの閾値は0.5、元の値のままがよい
		
		new_dat['score_sum'] = new_dat[   x_cols  ].sum(axis=1)
		
		# データセットを上書きする（なので、上でdat_evacuationを作成している）
		dat = new_dat.copy()
		
		###############################################################
		# モデルのロードと適用
		model_files = pd.DataFrame(model_files)
	#	x_vars = x_cols
		x_vars = ['score_sum']
		predict_probabilities = []
		for i in np.arange(0 , model_files.shape[0]) :
	#		if (i != 4): continue # 一旦特定のモデルだけにしておく
			model_file = list(model_files.iloc[i])[0]
	#		print(model_file)
			# joblibを使ってオブジェクトをロードし適用する。
			forest = joblib.load(model_file)
	#		predict_value = forest.predict( dat[ x_vars ] )
			predict_probability = forest.predict_proba( dat[ x_vars ] )[0][1]
			predict_probabilities.append(predict_probability)
			dat[ 'predict_probability' + "_" +str(i) ] = predict_probability
			dat[ 'applied_model' + "_" +str(i) ] = os.path.split(model_file)[1]
		# 複数モデルの単純平均
		predict_probability = np.mean(predict_probabilities)
		if (predict_probability >= 0.5 ) :
			predict_value = threshold
		else:
			predict_value = threshold -1
		###############################################################
		#
		
		output_file = re.sub("dataset","result_of_dataset" , os.path.split(input_file)[1] ) 
		output_file = re.sub("_estimate_", "_estimate_for_more_than" +str(threshold) + "_" , output_file )
		output_file = os.path.join(save_dir ,  output_file   )
		# 圧縮ファイルの保存に失敗することがあるので、（環境依存を避けるため）、非圧縮で保存する
		if (compress_flg == 0): output_file = re.sub(r'\.bz2$' , r'' , output_file)
		print(output_file)
		dat.to_csv(output_file ,  sep="\t" , index=False )
		
		###############################################
		print(r"眠気レベルの結果のファイルの保存直前")
		f = open( save_file ,'w') # 上書きモード
		f.write(dat['subject'][0] + '\t')
		f.write(str(dat['unixtime'][0]) + '\t')
		f.write(str(predict_value) + '\n')
		#f.write('\n')
		f.close()
		print(r"眠気レベルの結果のファイルの保存完了直後")
		
		return predict_value
		
	#except FovioError as e:
	#	print(r"fovio のデータの半分以上が欠損値なのでエラーにする")
	#	print(e)
	#	sys.exit(-1)
	#except RRiError as e:
	#	print(r"rri のデータの半分以上が欠損値なのでエラーにする")
	#	print(e)
	#	sys.exit(-2)
	except Exception as e: # 想定外のエラーをここで全てキャッチする
		print ("error")
		print(e)
		sys.exit(0)
	#	pass # 例外をキャッチしても特に何も処理を行わずにスルーしたい場合はpass文を使う。
	
	else : # 正常終了時の処理
	#	print('finish (no error)')
		print(r"正常終了")
	#	sys.exit(1)
	

#################################################################################################
#################################################################################################
#################################################################################################

if __name__ == "__main__":
	########################################
	"""
	home_dir=r"C:\DRMS"
	input_file = r"C:\Work\眠気推定_2019年度_多治見テストコース実車実験\data\dataset_with_diff.txt.bz2"
	input_file = r"C:\Temp\DRMS_estimate_data\dataset_for_estimate_1577067212.485.txt.bz2"
	input_file = r"C:\Temp\DRMS_estimate_data\dataset_for_estimate_*.txt.bz2"
	input_file = r"C:\DRMS\DRMS_estimate_data\dataset_for_estimate_*.txt.bz2"
	threshold=2
	threshold=3
	source_path = r"C:\Work\眠気推定_2019年度_簡易DS"
#	input_param = os.path.join( source_path , r'data' , 'クロス集計_眠気レベル' + str(threshold) +  '以上のパラメーター.txt.bz2' )
#	input_file = r"C:\Temp\DRMS_estimate_data\dataset_for_estimate_*.txt*.bz2"
	model_dir= home_dir
	save_dir =  r"C:\DRMS"
	"""
	########################################
	# パラメーターを受け取った時のみ対応する。
	if len(sys.argv) > 1:
		args = sys.argv
		home_dir =  args[1]
		input_file =  args[2]
		save_dir =  args[3]
		threshold =  args[4]
	########################################
	main(home_dir , input_file , save_dir , 2)
	main(home_dir , input_file , save_dir , 3)
