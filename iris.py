from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from cntk.device import try_set_default_device, gpu, cpu
import cntk as C
from cntk.learners import sgd
from cntk.logging import ProgressPrinter
from cntk.layers import Dense, Sequential

try_set_default_device(gpu(0))
#try_set_default_device(cpu())

# IRISデータを読み込む。
x, t = load_iris(return_X_y=True)

x_train_val, x_test, t_train_val, _t_test = train_test_split(x, t, test_size=0.3, random_state=0)

x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)

t_test = np.eye(3)[list(map(int,_t_test))] 

# 各種パラメータ定義
n_input = 4
n_hidden = 10
n_output = 3

# 学習データと正解ラベルの入力変数を定義する。
features = C.input_variable((n_input))
label = C.input_variable((n_output))

# ネットワークを構築する。
model = Sequential([
  Dense(n_hidden, activation=C.relu),
  Dense(n_hidden, activation=C.relu),
  Dense(n_hidden, activation=C.relu),
  Dense(n_output)])(features) 

ce = C.cross_entropy_with_softmax(model, label)
pe = C.classification_error(model, label)

minibatch = C.learning_parameter_schedule(0.125)
progress_printer = ProgressPrinter(0)
trainer = C.Trainer(model, (ce, pe), [sgd(model.parameters, lr=minibatch)],[progress_printer])

# 学習する。
n_epoch = 30
n_batchsize = 16 

for epoch in range(n_epoch):
  order = np.random.permutation(range(len(x_train)))
  
  aggregate_loss = 0.0
  
  for i in range(0, len(order), n_batchsize):
    index = order[i:i+n_batchsize]
    x_train_batch = x_train[index,:]
    _t_train_batch = t_train[index]
    t_train_batch = np.eye(3)[list(map(int,_t_train_batch))] 

    trainer.train_minibatch({features : x_train_batch, label : t_train_batch})
    sample_count = trainer.previous_minibatch_sample_count
    aggregate_loss += trainer.previous_minibatch_loss_average * sample_count

  last_avg_error = aggregate_loss / trainer.total_number_of_samples_seen

  avg_error = trainer.test_minibatch({features : x_test, label : t_test})
  print(' error rate on an unseen minibatch: {}'.format(avg_error))

# ONNX形式でモデルを保存する。
output_file_path = R"./cntk_iris_model.onnx"
model.save(output_file_path, format=C.ModelFormat.ONNX)

# ONNX形式で保存したモデルを読み込み、推論する。
print("========== START INFERENCE ==========")
reload_model = C.Function.load(output_file_path, device=C.device.gpu(0), format=C.ModelFormat.ONNX)

classifier = C.softmax(reload_model)

for i, train in enumerate(x_train):
  infer = np.argmax(classifier.eval([train]))
  print("[{}] in:{} correct:{} infer:{}".format(i,train, t_train[i], infer))

