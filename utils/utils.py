def assert_msg(condition, msg):
    if not condition:
        raise Exception(msg)
