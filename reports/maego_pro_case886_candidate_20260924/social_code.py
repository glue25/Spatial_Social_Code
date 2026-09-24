M.observe(1, "Shure", "Lucia says '确定没有,我感觉那一块儿就是一些烘焙原料'", source="spatial_trace", confidence=0.9, evidence_ids="S1")
M.belief(1, "Lucia", "the items in that area are baking ingredients", confidence=0.9, evidence="Lucia states she feels that area is just baking ingredients", evidence_ids="S1")
M.observe(2, "Shure", "Shure says '不,它那一小碗就是麻酱'", source="spatial_trace", confidence=0.9, evidence_ids="S2")
M.observe(4, "Shure", "Shure says '对,那个碗是我卸好的麻酱'", source="spatial_trace", confidence=0.9, evidence_ids="S4")
