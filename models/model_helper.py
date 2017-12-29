
def get_not_fc_para(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and k.find('classifier') == -1}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('loaded pretrained param~')
    return model