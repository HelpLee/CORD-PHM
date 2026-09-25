# Appendix B: architecture, optimization, and model selection

The shared architecture and system-specific interfaces are retained under
`../source_pretraining_and_transfer/shared_domain_components/`. The selected
multi-domain training configuration is
`../source_pretraining_and_transfer/selected_bearing_floor_cagrad/`; its direct
optimization dependencies are `component_gradient_routing/`,
`conditional_gradient_routing/`, and `transfer_mechanism_comparison/` in the
same parent directory.
