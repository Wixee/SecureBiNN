import numpy as np


def bits_to_dtype(bits):
    assert bits in {
        1, 8, 16, 32}, 'bits must in {1,8,16,32} but not {}'.format(bits)
    if bits == 1:
        return 'bool'
    else:
        return 'uint'+str(bits)


def get_random_mask(size, bits):
    return np.random.randint(low=0, high=2**bits, size=size, dtype=bits_to_dtype(bits))


class prf:

    def __init__(self, seed):
        self.seed = seed
        self._mod = 1000000009

    def __get_prf_inputs(self, cnter):
        return (self.seed * cnter ^ cnter) * cnter

    def prf_uint(self, bits, cnter, size):
        prf_inputs = self.__get_prf_inputs(cnter)
        np.random.seed(prf_inputs % self._mod)
        return np.random.randint(low=0, high=2**bits, size=size, dtype=bits_to_dtype(bits))


class RandomHandler:

    def __init__(self, role, seed_role, seed_role_p1, cnter=0, cnter_step=100):

        # role
        self.role = role

        # set seed(role), seed(role+1)
        self.seed_role = seed_role
        self.seed_role_p1 = seed_role_p1

        # set cnter
        self.cnter = cnter
        self.cnter_step = cnter_step

        # set prf
        self.prf_role = prf(self.seed_role)
        self.prf_role_p1 = prf(self.seed_role_p1)

    def step_synchronous(self):
        self.cnter = ((self.cnter//self.cnter_step + 1)
                      * self.cnter_step) % 1000000

    def get_3_out_of_3_shares(self, bits, size):

        # generate outputs
        output_with_seed_role_p1 = self.prf_role_p1.prf_uint(
            bits=bits, cnter=self.cnter, size=size)
        output_with_seed_role = self.prf_role.prf_uint(
            bits=bits, cnter=self.cnter, size=size)

        # self.cnter += 1

        if bits == 1:
            return output_with_seed_role_p1 ^ output_with_seed_role
        else:
            return output_with_seed_role_p1 - output_with_seed_role

    def get_same_random_with(self, target_role, bits, size):
        '''
        role_0 has (seed_0, seed_1)
        role_1 has (seed_1, seed_2)
        role_2 has (seed_2, seed_0)
        '''
        ret = None
        if target_role == (self.role + 1) % 3:
            ret = self.prf_role_p1.prf_uint(
                bits=bits, cnter=self.cnter, size=size)
        elif target_role == (self.role - 1) % 3:
            ret = self.prf_role.prf_uint(
                bits=bits, cnter=self.cnter, size=size)
        else:
            raise Exception('target role is not correct')
        # self.cnter += 1
        return ret
