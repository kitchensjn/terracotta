![Logo](../assets/20241202/logo.png)

Document last updated 2025-04-08.

## Overview

`terracotta` is a belief propagation method for estimating migration surfaces and the locations of genetic ancestors from gene trees and/or small ancestral recombination graphs. The genetic relationships between samples inform us about their spatial history. If we know that two samples have a most recent common ancestor 50 generations in the past, then we also know that those samples must have been in the same location at that time. Inspired by EEMS and MAPS from the Novembre Lab, `terracotta` discretizes space into many tiles and uses a stepping stone movement model where individuals can migrate between neighboring tiles. The rates of migration are stored within a transition matrix; when plotted, this transition matrix reflects the migration surface of the system.

## Simulations