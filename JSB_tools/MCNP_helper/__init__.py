from JSB_tools.MCNP_helper.geometry.geom_core import Cell, Surface, MCNPNumberMapping, TRCL
from JSB_tools.MCNP_helper.geometry.primitives import RightCylinderSurface, RightCylinder, CuboidSurface, CellGroup, \
     CuboidCell
from JSB_tools.MCNP_helper.input_deck import InputDeck
from JSB_tools.MCNP_helper.outp_reader import OutP, StoppingPowerData
from JSB_tools.MCNP_helper.input_deck import F4Tally, TallyBase
import JSB_tools.MCNP_helper.units
from typing import Union
__all__ = []
from JSB_tools import trace_prints

# trace_prints()

# Todo:
#  - Reorganize module structure. It is a mess. Merge geom_core, __geom_helpers, and primitives into two modules.
#  - Redo tally_n reader. It's too complex.
#  - Test new Sphere primitive.
#  - Make a module for Tallys
#


class CylFMESH(TallyBase):
    """Cylindrical Fmesh for MCNP"""

    # all_meshes = MCNPNumberMapping('CylFMESH', 1)
    # all_meshes = F4Tally.all_f4_tallies
    def __init__(self, particle: str,
                 rmaxs,
                 axis_lengths,
                 origin=(0, 0, 0),
                 rbins: Union[int, tuple] = 10,
                 axis_bins: Union[int, tuple] = 10, axs_hat=(0, 0, 1),
                 theta_bins=1, theta_maxs=1,
                 radius_hat=(1, 0, 0), tally_number=None, fmesh_name=None, ref=None):
        """


        Args:
            particle:
            rmaxs:
            axis_lengths: Distance along axis specified by `axs`
            origin: max theta in revolutions
            rbins: Number of radial bins.
            axis_bins: Number of bins along the axis
            axs_hat:
            radius_hat:
            tally_number:
            fmesh_name:
            theta_bins: Number of bins in theta. Pretty much always should be 1.
            theta_maxs:
            ref: REF keyword for mesh weight windows
        """
        super(CylFMESH, self).__init__()

        self.tally_number = tally_number
        if self.tally_number is not None:
            assert str(self.tally_number)[-1] == '4', 'F4 tally_n number must end in a "4"'

        self.__name__ = fmesh_name

        self.rmax = rmaxs if hasattr(rmaxs, '__iter__') else (rmaxs,)
        self.rbins = rbins if hasattr(rbins, '__iter__') else (rbins,)

        self.axis_length = axis_lengths if hasattr(axis_lengths, '__iter__') else (axis_lengths,)
        self.axis_bins = axis_bins if hasattr(axis_bins, '__iter__') else (axis_bins,)

        self.theta_max = theta_maxs if hasattr(theta_maxs, '__iter__') else (theta_maxs,)
        self.theta_bins = theta_bins if hasattr(theta_bins, '__iter__') else (theta_bins,)

        self.origin = origin
        self.particle = particle
        self.axs_hat = axs_hat
        self.radius_hat = radius_hat

        self.all_f4_tallies[self.tally_number] = self

        self.ref = ref

    @property
    def fmesh_number(self):
        return int(str(self.tally_number) + '4')

    @property
    def tally_card(self):
        return self.__repr__()

    def __repr__(self):
        def f(a):  # iter to MCNP input
            return " ".join(map(str, a))
        optional = {}  # for any optional kwargs. None implemented yet...
        optional = ' '.join(f'{k}={v}' for k, v in optional.items())
        out = f'FMESH{self.fmesh_number}:{self.particle} GEOM=cyl ORIGIN={f(self.origin)} AXS={f(self.axs_hat)} ' \
              f'VEC={f(self.radius_hat)} {optional} $ {self.__name__}\n' \
              f'     IMESH {f(self.rmax)}  IINTS {f(self.rbins)}\n' \
              f'     JMESH {f(self.axis_length)}  JINTS {f(self.axis_bins)}\n' \
              f'     KMESH {f(self.theta_max)}  KINTS {f(self.theta_bins)}'
        return out


if __name__ == '__main__':
    fmesh = CylFMESH('p', rmax=10, rbins=2, axis_length=5, theta_max=1, origin=[0, 0, 1])
    fmesh2 = CylFMESH('p', rmax=100, rbins=2, axis_length=5, theta_max=1, origin=[0, 0, 1])
    print(fmesh)
    print(fmesh2)