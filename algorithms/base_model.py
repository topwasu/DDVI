import logging
import time
from abc import ABC, abstractmethod


log = logging.getLogger(__name__)


class BaseModel(ABC):
    @abstractmethod
    def _call_schedulers(self):
        pass

    def train(self, dataloader, training_routine=lambda *args, **kwargs: None):
        log.info("Start training")
        start = time.time()
        self.start_time = start
        for epoch in range(self.num_epochs):
            if epoch % 5 == 0:
                log.info(f'Epoch: {epoch}')
            self.ep = epoch
            st = time.time()
            res = self._train_epoch(dataloader)
            self._call_schedulers()

            if epoch % 5 == 0:
                log.info(f'Training time = {(time.time()- st)/len(dataloader)}')
                log.info(f'Current time {time.time() - start}')
                log.info(f'Losses {res}')
                    
            training_routine(self.save_folder, self, epoch, self.num_epochs)

        end = time.time()
        log.info('End training - training time: {} seconds'.format(end - start))

    @abstractmethod
    def _train_epoch(self, dataloader):
        pass

    @abstractmethod
    def eval(self, dataloader):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass