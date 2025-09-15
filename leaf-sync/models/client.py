import random
import warnings


class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data

    # def train(self, num_epochs=1, batch_size=10, minibatch=None):
    #     """Trains on self.model using the client's train_data.

    #     Args:
    #         num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
    #         batch_size: Size of training batches.
    #         minibatch: fraction of client's data to apply minibatch sgd,
    #             None to use FedAvg
    #     Return:
    #         comp: number of FLOPs executed in training process
    #         num_samples: number of samples used in training
    #         update: set of weights
    #         update_size: number of bytes in update
    #     """
    #     if minibatch is None:
    #         data = self.train_data
    #         comp, update = self.model.train(data, num_epochs, batch_size)
    #     else:
    #         frac = min(1.0, minibatch)
    #         num_data = max(1, int(frac*len(self.train_data["x"])))
    #         xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
    #         data = {'x': xs, 'y': ys}

    #         # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
    #         num_epochs = 1

    #         print("Client %s: training on %d out of %d samples" % (self.id, num_data, len(self.train_data["x"])))
    #         comp, update = self.model.train(data, num_epochs, num_data)
    #     num_train_samples = len(data['y'])
    #     return comp, num_train_samples, update


    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        """
        Treina no train_data.
        - FedAvg: usa num_epochs e batch_size (full-batch se batch_size<=0 ou >=n).
        - Minibatch: usa fração dos dados; se frac>=1.0, usa dataset completo sem amostrar e faz um único passo full-batch.
        Retorna: comp, num_train_samples, update
        """
        x_all, y_all = self.train_data['x'], self.train_data['y']
        n = len(y_all)

        if minibatch is None:
            # ----- FedAvg -----
            B = batch_size
            if B is None or B <= 0 or B >= n:
                B = n  # full-batch para equivalência com Minibatch=100%
            data = {'x': x_all, 'y': y_all}
            E = num_epochs
            comp, update = self.model.train(data, E, B)
            num_train_samples = len(data['y'])
            return comp, num_train_samples, update

        else:
            # ----- Minibatch SGD -----
            frac = float(min(1.0, minibatch))
            num_data = max(1, int(frac * n))

            if num_data >= n:
                # 100%: usa TODO o dataset, sem amostrar/embaralhar
                xs, ys = x_all, y_all
            else:
                # fração < 100%: amostra sem reposição (pode usar numpy para seed por round)
                idx = random.sample(range(n), num_data)
                xs = [x_all[i] for i in idx]
                ys = [y_all[i] for i in idx]

            data = {'x': xs, 'y': ys}

            # Minibatch treina 1 época; para equivalência, um único passo full-batch
            E = 1
            B = len(xs)  # full-batch da fração selecionada
            comp, update = self.model.train(data, E, B)
            num_train_samples = len(data['y'])
            return comp, num_train_samples, update

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
