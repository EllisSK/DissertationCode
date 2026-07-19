# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Physical constants and flume/barrier geometry shared across the analysis and solver packages.

All lengths are in metres unless stated otherwise. The barrier is described by a
"gap1-gap2-gap3" setup string giving the three gap sizes in millimetres, interleaved
with the three fixed planks from the bed upwards:

    bed | gap1 | plank1 (0.2) | gap2 | plank2 (0.1) | gap3 | plank3 (0.1)
"""

# Standard gravitational acceleration (m/s^2) used by the discharge models
GRAVITY = 9.80665

# Barrier plank heights from bottom to top (m)
PLANK_1_HEIGHT = 0.2
PLANK_2_HEIGHT = 0.1
PLANK_3_HEIGHT = 0.1

# Flume geometry (m)
FLUME_LENGTH = 12.5
BARRIER_POSITION = 5.0

# Channel width (m) assumed by the wetted-perimeter relations (P = width + 2h).
# The discharge relations are expressed per metre width of channel.
CHANNEL_WIDTH = 1.0

# Diameter (m) of the pipe on which the electromagnetic flow meter is mounted,
# used to convert the meter's velocity uncertainty into a flow-rate uncertainty
FLOW_METER_PIPE_DIAMETER = 0.350
