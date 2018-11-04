import constants


def parse_config():
    with open(constants.YOLO_CONFIG_FILE) as f:
        file_content = f.read()
        config, config_dict = [], None
        for line in file_content.split('\n'):
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            if line.startswith('['):
                if config_dict is not None:
                    config.append(config_dict)
                config_dict = dict()
                config_dict['type'] = line[1:-1]
            else:
                if config_dict is None:
                    raise Exception("Got None config_dict")
                parameter, value = line.split('=')
                parameter, value = parameter.strip(), value.strip()
                config_dict[parameter] = value
    return config
