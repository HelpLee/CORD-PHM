"""Milling-only control: identical initialization, then remove unused domains."""
from reference_joint_model import JointModel as ReferenceModel

DOMAINS = ('milling',)

class JointModel(ReferenceModel):
    def __init__(self):
        super().__init__()
        # Instantiate identically before pruning so shared/milling tensors and
        # the random-number state exactly match the three-domain control.
        for modules in (self.encoder.stems, self.encoder.adapters,
                        self.channels, self.decoders):
            for domain in ('bearing', 'battery'):
                del modules[domain]
