class BaseParameter:
    def __init__(self, name, dim, is_complex=False):
        """

        :param name:
        :param dim:
        :param is_complex:
        """
        self.name = name
        self.dim_complex = dim
        self.dim_real = (1 + int(is_complex)) * dim
        self.is_complex = is_complex

    @property
    def dim(self):
        return self.dim_real

    def parameter_range(self, n_steps, **kwargs):
        raise NotImplemented

    def random_sample(self):
        raise NotImplemented
