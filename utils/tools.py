import torch


def tocuda(data):
    # convert tensor data in dictionary to cuda when it is a tensor
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()

    return data


def print_notification(content_list, notification_type='NOTIFICATION'):
    print('---------------------- {0} ----------------------'.format(notification_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')


def safe_load_weights(model, saved_weights):
    try:
        model.load_state_dict(saved_weights)
    except RuntimeError:
        try:
            weights = saved_weights
            weights = {k.replace('module.', ''): v for k, v in weights.items()}
            model.load_state_dict(weights)
        except RuntimeError:
            try:
                weights = saved_weights
                weights = {'module.' + k: v for k, v in weights.items()}
                model.load_state_dict(weights)
            except RuntimeError:
                try:
                    pretrained_dict = saved_weights
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape))}
                    assert len(pretrained_dict) != 0
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    non_match_keys = set(model.state_dict().keys()) - set(pretrained_dict.keys())
                    notification = []
                    notification += ['pretrained weights PARTIALLY loaded, following are missing:']
                    notification += [str(non_match_keys)]
                    print_notification(notification, 'WARNING')
                except Exception as e:
                    print(f'pretrained weights loading failed {e}')
                    exit()
    print('weights safely loaded')