# BigData 项目笔记

先考虑简单版本，假设海水密度 $\rho$ 和比热容 $C_p$ 都是常数，则要计算 $OHC(x, y, z, t) [J]$，可以利用公式：
$$
OHC(x,y,z,t) = \rho \cdot C_p \cdot V(x,y,z) \cdot T(x,y,z,t)
$$
 则关键是计算 $V(x,y,z)$，可以假设每个区域为矩形，则 $V(x,y,z) = A(lat) \cdot dz(z)$

根据地球半径和纬度（latitude）可以计算出随着纬度变化的表面积 $A(lat)[m^2]$，根据 $deapths$，可以计算 $dz$，第一个值从海面开始计数，最后一个值到 $deapths$ 的最后一个值截止，中间的值取前后两个值的中间值。

