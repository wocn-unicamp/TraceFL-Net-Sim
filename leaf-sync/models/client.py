import random
import warnings


class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train(data, num_epochs, batch_size)
        # else:
        #     frac = min(1.0, minibatch)
        #     num_data = max(1, int(frac*len(self.train_data["x"])))
        #     xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
        #     data = {'x': xs, 'y': ys}

        #     # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
        #     num_epochs = 1
        #     comp, update = self.model.train(data, num_epochs, num_data)

        else:
            # ----- Minibatch SGD -----
            x_all, y_all = self.train_data['x'], self.train_data['y']
            n = len(y_all)
            frac = float(min(1.0, minibatch))
            n = len(y_all)
            num_data = max(1, int(round(frac * n)))

            # Amostra sem reposição quando frac < 1.0; para 100% usa todos
            # (use NumPy para facilitar tornar determinístico se quiser)
            import numpy as np
            if num_data >= n:
                idx = np.arange(n)
            else:
                idx = np.random.choice(n, size=num_data, replace=False)


            xs = [x_all[i] for i in idx]
            ys = [y_all[i] for i in idx]
            data = {'x': xs, 'y': ys}

            # >>> AQUI ESTÁ A MUDANÇA-CHAVE: usar o MESMO batch_size do FedAvg <<<
            # - Se batch_size não for passado (None ou <=0), caia para o padrão (ex.: 10)
            B = batch_size if (batch_size is not None and batch_size > 0) else 10
            # - Garanta que não ultrapasse o tamanho do subconjunto selecionado
            B = min(B, len(xs))

            E = 1  # uma época no modo minibatch
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
