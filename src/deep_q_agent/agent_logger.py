import os
import pickle


class DeepQLog(object):
    def __init__(self, log_threshold = 1, log_path="agent_log.csv"):
        self.rows = []
        self.labels = None
        self.log_threshold = log_threshold
        self.log_path = log_path
        self.log_file_reference = None

    def __call__(self, col_labels, values, **kwargs):
        """[summary]

        Args:
            log_name ([type]): [description]
        """

        # Initialize our labels
        if self.labels is None:
            self.labels = col_labels
            with open(self.log_path, "w") as f:
                f.write(",".join(self.labels) + "\n")
            


        # Now check (for future calls) if the labels match
        assert self.labels == col_labels, f"Label Do Not Match: {self.labels} != {col_labels}"

        # Now add the values to our rows
        self.rows.append(values)

        # If we have filled the values enough, write and clear
        if len(values) >= self.log_threshold:
            with open(self.log_path, "a") as f:
                for r in self.rows:
                    r = [str(v) for v in r]
                    f.write(",".join(r) + "\n")
            self.rows.clear()

