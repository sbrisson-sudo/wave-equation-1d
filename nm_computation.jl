module NM1d

# IMPORTATIONS

import Interpolations: LinearInterpolation

using Printf

using TimerOutputs

# EXPORTATIONS

export eigen_frequencies_homogene, eigen_function_at_surface_homogene, eigen_function_homogene
export compute_mode_surface, compute_eigen_frequencies, compute_mode

# procedures

function compute_mode_surface(ρ::T, λ::T, ω::T, αmax::T, L::T ; debug=false) where T <: AbstractFloat

    N::Int = max(div(L, αmax/ω/4), 100)
    h::T = L/N
    X = collect(range(0.0, L, length = 2*N))

    debug && println("# Performing Runge-Kutta with $N steps.")

    λVec = [λ for x in X]
    ρVec = [ρ for x in X]

    S12 = 1 ./λVec
    S21 = -ρVec.*ω^2

    y1::T = 1.0 # displacement
    y2::T = 0.0 # traction

    for i = 1:N-1

        a1 = h * S12[2*i]*y2
        a2 = h * S21[2*i]*y1

        b1 = h * S12[2*i+1]*(y2+a2/2) 
        b2 = h * S21[2*i+1]*(y1+a1/2) 

        c1 = h * S12[2*i+1]*(y2+b2/2) 
        c2 = h * S21[2*i+1]*(y1+b1/2) 

        d1 = h * S12[2*(i+1)]*(y2+c2) 
        d2 = h * S21[2*(i+1)]*(y1+c1)
        
        y1 += a1/6 + b1/3 + c1/3 + d1/6
        y2 += a2/6 + b2/3 + c2/3 + d2/6

    end

    debug && println("[compute_mode_surface] ω=$ω : us = $y1, ts = $y2")

    return y1, y2

end

function compute_mode(ρ::T, λ::T, ω::T, αmax::T, L::T ; debug=false) where T <: AbstractFloat

    N::Int = max(div(L, αmax/ω/4), 100)
    h::T = L/N
    X = collect(range(0.0, L, length = 2*N))

    debug && println("# Performing Runge-Kutta with $N steps.")

    λVec = [λ for x in X]
    ρVec = [ρ for x in X]

    S12 = 1 ./λVec
    S21 = -ρVec.*ω^2

    y1 = ones(T,N) # displacement
    y2 = zeros(T,N) # traction

    for i = 1:N-1

        a1 = h * S12[2*i]*y2[i]
        a2 = h * S21[2*i]*y1[i]

        b1 = h * S12[2*i+1]*(y2[i]+a2/2) 
        b2 = h * S21[2*i+1]*(y1[i]+a1/2) 

        c1 = h * S12[2*i+1]*(y2[i]+b2/2) 
        c2 = h * S21[2*i+1]*(y1[i]+b1/2) 

        d1 = h * S12[2*(i+1)]*(y2[i]+c2) 
        d2 = h * S21[2*(i+1)]*(y1[i]+c1)
        
        y1[i+1] = y1[i] + a1/6 + b1/3 + c1/3 + d1/6
        y2[i+1] = y2[i] + a2/6 + b2/3 + c2/3 + d2/6

    end

    return y1, y2

end

function compute_eigen_frequencies(ρ::T, λ::T, L::T, ω1::T, ω2::T; ε = 1e-2, debug = false) where T <: AbstractFloat

    # finding the maximum velocity (needed to compute the integration step)
    αmax = sqrt(λ/ρ)
    
    # initial regular sampling to frame the zeros
    dω = (ω2 - ω1)/1000  
    Ω = collect(range(ω1, ω2, step = dω))
    Ωbounds = []

    debug && println("Searching for eigen frequencies bounds.")

    T0a = compute_mode_surface(ρ, λ, ω1, αmax, L, debug=debug)[2]
    for (i,ω) in enumerate(Ω[2:end])
        T0b = compute_mode_surface(ρ, λ, ω, αmax, L, debug=debug)[2]
        T0b * T0a < 0 && push!(Ωbounds, (Ω[i], ω))
        T0a = T0b
    end

    Nfp = length(Ωbounds)
    debug && println("At least $Nfp eigen frequencies to find.")

    # finding all eigenfrequencies by dichotomy
    eigen_freq = zeros(Float64, Nfp)

    for (ifp, (ωa, ωb)) in zip(1:Nfp, Ωbounds)

        debug && println(@sprintf("\nSearching eigen frequency between %.4g and  %.4g", ωa, ωb))

        T0a::Float64 = compute_mode_surface(ρ, λ, ωa, αmax, L, debug=debug)[2]
        T0b::Float64 = compute_mode_surface(ρ, λ, ωb, αmax, L, debug=debug)[2]

        ωc::Float64 = ωb
        T0c::Float64 = T0b
        
        nit = 0

        # on cherche un zéro de T(omega)
        while abs(T0c) > ε
            nit += 1
            ωc = (ωa + ωb)/2
            T0c =  compute_mode_surface(ρ, λ, ωc, αmax, L, debug=debug)[2][end]
            if T0c*T0a < 0
                ωb = ωc
                T0b = T0c
            else
                ωa = ωc
                T0a = T0c
            end
        end

        eigen_freq[ifp] = ωc
        debug && println(@sprintf("Find a normal mode at ω = %.4g, %d iterations.", ωc, nit))
    end

    return eigen_freq
end


function eigen_frequencies_homogene(ρ, λ, L, ω1, ω2)
    α = sqrt(λ/ρ)
    ω0 = π*α/L
    nmin = div(ω1,ω0)
    nmax = div(ω2,ω0)
    return [ω0*n for n=nmin:nmax]
end 

function eigen_function_homogene(ρ, λ, ω, L; Npt=100)
    α = sqrt(λ/ρ)
    x = range(0.0, L, length=Npt)
    u = (1/√(ρ*L)).*cos.(ω/α.*x) # displacement
    t = (-√λ*ω/L).*sin.(ω/α.*x)  # traction
    return u,t
end 

function eigen_function_at_surface_homogene(ρ, λ, ω, L)
    α = sqrt(λ/ρ)
    u = 1/√(ρ*L)*cos(ω*L/α) # displacement
    t = -√λ*ω/L*sin(ω*L/α)  # traction
    return u,t
end


end # module


if abspath(PROGRAM_FILE) == @__FILE__

    using .NM1d

    L = 1.0

    ρ = 1.0
    λ = 1.0

    ω1 = 1e-5
    ω2 = 500.0

    ωn = compute_eigen_frequencies(ρ, λ, L, ω1, ω2)

    print(ωn)


end