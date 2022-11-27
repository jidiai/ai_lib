import time
from .table import Table
from utils.logger import Logger
import threading


class DataServer:
    def __init__(self, id, cfg):
        self.id = id
        self.cfg = cfg
        self.tables = {}

        self.table_lock = threading.Lock()

        self.read_timeout = self.cfg.read_timeout
        self.table_cfg = self.cfg.table_cfg

        Logger.info("{} initialized".format(self.id))

    def create_table(self, table_name):
        with self.table_lock:
            Logger.warning("table_cfgs:{} uses {}".format(self.id, self.table_cfg))
            self.tables[table_name] = Table(self.table_cfg)
            Logger.info("{} created data table {}".format(self.id, table_name))

    def remove_table(self, table_name):
        with self.table_lock:
            if table_name in self.tables:
                self.tables.pop(table_name)
            Logger.info("{} removed data table {}".format(self.id, table_name))

    def get_statistics(self, table_name):
        try:
            with self.table_lock:
                statistics = self.tables[table_name].get_statistics()
            return statistics
        except KeyError:
            time.sleep(1)
            info = "{}::get_table_stats: table {} is not found".format(
                self.id, table_name
            )
            Logger.warning(info)
            return {}

    def save(self, table_name, data):
        try:
            with self.table_lock:
                table: Table = self.tables[table_name]
            if len(data) > 0:
                table.write(data)
        except KeyError:
            time.sleep(1)
            Logger.warning(
                "{}::save_data: table for {} is not found".format(self.id, table_name)
            )

    def sample(self, table_name, batch_size, wait=False):
        try:
            with self.table_lock:
                table: Table = self.tables[table_name]
            samples = None
            samples = table.read(batch_size, timeout=self.read_timeout)
            if samples is not None:
                return samples, True
            else:
                return samples, False
        except KeyError:
            Logger.warning(
                "{}::sample_data: table {} is not found".format(self.id, table_name)
            )
            time.sleep(1)
            samples = None
            return samples, False

    def load_data(self, table_name, data_path):
        """
        TODO(jh): maybe support more data format?
        data now are supposed to be stored in pickle format as a list of samples.
        """
        # check extension
        assert data_path[-4:] == ".pkl"

        # get table
        with self.table_lock:
            table: Table = self.tables[table_name]

        # load data from disk
        import pickle as pkl

        with open(data_path, "rb") as f:
            samples = pkl.load(f)

        assert (
            len(samples) <= table.capacity
        ), "load too much data(size{}) to fit into table(capacity:{})".format(
            len(samples), table.capacity
        )

        # write samples to table
        table.write(samples)
        Logger.info(
            "Table {} load {} data from {}".format(self.id, len(samples), data_path)
        )
