class DeepQLog(object):
    def __init__(self):
        self.__core_log = {}

    def __call__(self, log_name, **kwargs):
        """[summary]

        Args:
            log_name ([type]): [description]
        """
        # If the log name already exists, we update the values
        if log_name not in self.__core_log:
            self.__core_log[log_name] = {}

        # Iterate over the kwargs
        for key in kwargs:
            # Grab the item
            item = kwargs[key]
            # Check to see if the passed value is in the log
            if key in self.__core_log[log_name]:
                self.__core_log[log_name][key].append(item)
            else:
                self.__core_log[log_name][key] = [item]

    def __repr__(self) -> str:
        display = ""
        for log_entry in self.__core_log:
            display += f"*{log_entry}*\n"
            for sub_entry in self.__core_log[log_entry]:
                display += f"   -> {sub_entry}: {self.__core_log[log_entry][sub_entry]}\n"
            display += "\n"
        return display