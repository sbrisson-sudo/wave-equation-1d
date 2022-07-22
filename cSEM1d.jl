
using FFTW

# 0). Input parameters

L = 1.0
ρ = 1.0
λ = 1.0

t_length = 2.0 # time for which
nb_pt_dtn = 1000
dt = t_length/nb_pt_dtn

ω1 = 1e-5
ω2 = 500.0

ε = 1e-3 # precision at which we want to cancel the traction at surface

# 1). Computing eigen-frequencies

include("nm_computation.jl")
using .NM1d 

Ωn = compute_eigen_frequencies(ρ, λ, L, ω1, ω2, ε=ε)
nb_ef = length(Ωn)

# 2). Choosing frequencies at which computing the DtN operator

ε_ef = 1e-2 # faire que ce soit indépendant du système étudié
Ω = collect(range(ω1, ω2, length = nb_pt_dtn))

# We displace the points too close from an eigen frequency
for (i,ω) in enumerate(Ω)
    for ωn in Ωn
        if abs(ω-ωn) < ε_ef
            ω > ωn && Ω[i] += ε_ef
            ω <= ωn && Ω[i] -= ε_ef
        end
    end
end

# 3). Computing the DtN operator in frequency

# 3.a). Computing the DtN operator main term

A_f = zeros(Float64, nb_pt_dtn)

for i=1:nb_pt_dtn
    u,t = compute_mode(ρ, λ, Ω[i], √(λ/ρ), L)
    A_f[i] = u/t
end 

# 3.b). Computing the residuals at the eigen-frequencies

int_I1(ρ, u) = ρ/length(U) * sum(u.^2)    

A_res = zeros(Float64, nb_ef)
for i=1:nb_ef
    u,t = compute_mode(ρ, λ, Ωn[i],√(λ/ρ), L)
    A_res[i] = -t[end]^2/(2*Ωn[i]*int_I1(ρ,u))
end

A_f_res = zeros(Float64, nb_ef)
for i=1:nb_ef
    A_f_res[i] = A_f[i] - sum([A_res[n]/(Ω[i]-Ωn[n]) for n=1:nb_ef])
end

# 3.c). Computing the regularization term 

C_f = zeros(Float64, nb_pt_dtn)

c0 = √(λ*ρ) # attention : implémenter les autres termes dans le cas d'un milieu non homogène

for (i,ω) in enumerate(Ω)
    C_f[i] = c0*ω*1im
end 

A_f_res_reg = A_f_res .- C_f 

# 4). Computing the DtN operator in time

A_t_res_reg = fft(A_f_res_reg)

A_t_cplx = zeros(Float64, nb_pt_dtn)
A_t = zeros(Float64, nb_pt_dtn)

for i=1:nb_pt_dtn
    t = dt*i
    A_t_cplx[i] = sum([A_res[n]*1im*exp(1im*Ωn[n]*t) for n=1:nb_ef]) + A_t_res_reg[i]
end

A_t = real(A_t_cplx)

# 5). Writting the DtN operator and the regularization term