from modelarts.session import Session
from modelarts.model import Model
from modelarts.config.model_config import ServiceConfig

session = Session(config_file='user.json')
model_instance = Model(session, model_id="6450c8e4-d77f-4f33-90bc-5577fde04701")
configs = [ServiceConfig(model_id=model_instance.model_id, weight="100", specification="local", instance_count=1)]
predictor_instance = model_instance.deploy_predictor(configs=configs)
