class InputUnit:

    def __init__(self, obj, value):
        self.obj = obj
        self.expected_value = value

    # replacement for toString
    def __repr__(self):
        return "true output: %d; %s" % (self.expected_value, str(self.obj))
