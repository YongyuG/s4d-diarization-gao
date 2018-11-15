import logging
import numpy as np


class GaussFull:
    """
    Class to model a speaker by a gaussian with full covariance
    """
    def __init__(self, name, dim):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.count = 0
        self.dim = dim
        self.stat1 = np.zeros(dim)
        self.stat2 = np.zeros((dim, dim))
        self.cov_log_det = 0;
        self.mu = None
        self.mu_dot = np.NAN
        self.cov = None
        self.partial_bic = np.NaN

    def add(self, features):
        """
        Accumulate statistics for *features*
        :param features: numpy.ndarray

        """
        self.count += features.shape[0]  # add the number of features
        self.stat1 += features.sum(axis=0)
        self.stat2 += np.dot(features.T, features)

    def _cov_log_det(self):
        """
        Compute the log det of the covariance matrix
        :return:  float
        """
        s, d = np.linalg.slogdet(self.cov)
        return d
        # cov_chol, lower = scipy.linalg.cho_factor(self.cov)
        # return 2.0 * numpy.sum(numpy.log(numpy.diagonal(cov_chol)))

    def compute(self):
        """
        Compute the mean and covariance according the statistique, the log det
         of the covariance and the partial BIC :math:`PBIC`.

        :math:`PBIC_{x}  = \\frac{n_x}{2} \\log|\\Sigma_x|`

        """
        self.mu = self.stat1 / self.count
        tmp = self.mu[:, np.newaxis]
        self.mu_dot = np.dot(tmp, tmp.T)
        self.cov = self.stat2 / self.count - self.mu_dot
        self.cov_log_det = self._cov_log_det()
        self.partial_bic = self.cov_log_det * 0.5 * self.count

    @classmethod
    def merge(cls, m1, m2):
        """
        Merge two models *m1* and *m2*. Compute the new mean (*mu*),
        covariance (*cov*) and PBIC *partial_bic*

        :param m1: a GaussFull object
        :param m2: a GaussFull object
        :return: a GaussFull object
        """
        m = GaussFull(m1.name, m1.dim)
        m.count = m1.count + m2.count
        m.stat1 = m1.stat1 + m2.stat1
        m.stat2 = m1.stat2 + m2.stat2
        m.mu = m.stat1 / m.count
        tmp = m.mu[:, np.newaxis]
        m.cov = m.stat2 / m.count - np.dot(tmp, tmp.T)
        m.cov_log_det = m._cov_log_det()
        m.partial_bic = m.cov_log_det * 0.5 * m.count
        return m

    @classmethod
    def merge_partial_bic(cls, m1, m2):
        """
        Merge statistic accumulators of two a GaussFull objects, compute
        the new log det.

        :param m1: a GaussFull object
        :param m2: a GaussFull object
        :return: the log det
        """
        try:
            count = m1.count + m2.count
            mu = ((m1.stat1 + m2.stat1) / count)[:, np.newaxis]
            cov = (m1.stat2 + m2.stat2) / count - np.dot(mu, mu.T)
            s, d = np.linalg.slogdet(cov)
            # cov_chol, lower = scipy.linalg.cho_factor(cov, overwrite_a=True, check_finite=False)
            # d = 2.0 * numpy.sum(numpy.log(numpy.diagonal(cov_chol)))
            d *= 0.5 * count
            return d
        except:
            logging.warning('Det problem set to NaN ', m1.name, m2.name)
            return np.nan

    @classmethod
    def cst_bic(cls, dim, alpha):
        """
        Compute the BIC constant:

            :math:`cst  = \\frac{1}{2} \\alpha \\left(d + \\frac{d(d+1)}{2}\\right)`

        where :math:`d`is the feature dimension (*dim*)
        and :math:`alpha` a threshold (*alpha*)

        :param dim: the feature dimension
        :param alpha: the threshold
        :return: the constant
        """
        return 0.5 * alpha * (dim + (0.5 * ((dim + 1) * dim)))




