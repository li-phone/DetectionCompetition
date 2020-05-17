from modelarts.session import Session
from modelarts.model import Model
from modelarts.config.model_config import ServiceConfig
session = Session('user.json')
model_instance = Model(session, model_id="c6916c43-e67b-458d-a692-42b1a87cc43d")
configs = [ServiceConfig(model_id=model_instance.model_id, weight="100",specification="local",instance_count=1)]
predictor_instance = model_instance.deploy_predictor(configs=configs)                         # 部署为本地服务Predictor
data_path='input/img_1.jpg'
predict_result = predictor_instance.predict(data=data_path, data_type='files')     # 本地推理预测
print(predict_result)
