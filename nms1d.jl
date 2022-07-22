module NMS1d

# IMPORTATIONS

import PyPlot
const plt = PyPlot
import PyCall: pyimport
const mpl = pyimport("matplotlib")
mpl.style.use("myStyle")

import DSP: conv

import Interpolations: LinearInterpolation

using Printf

using TimerOutputs

# EXPORTATIONS

export computeMode, computeNormalModes, NMS
export computeNormalModes_bm, NMS_bm
export source_ricker, source_heaviside
export plotField, plotFrequencies, plotModes, plotNMSresults, plotSource

# procedures


"""
    Integrate the displacement and traction using RK4 starting from u=1.0, t=0.0 at the based

    INPUTS
        ρ {Float} : density
        λ {Float} : elastic modulus
        ω {Float} : frequency of the mode
        αmax {Float} : maximum wave celerity
        L {FLoat} : bar length
        debug = false : verbosity
    
    OUTPUTS
        (u,t) {Tuple{Vector{Float}}} : displacement and traction vectors

"""
function computeMode(ρ::Float64, λ::Float64, ω::Float64, αmax::Float64, L::Float64 ; debug=false)

    N::Int = max(div(L, αmax/ω/4), 100)
    h::Float64 = L/N
    X = collect(range(0.0, L, length = 2*N))

    debug && println("# Performing Runge-Kutta with $N steps.")

    λVec = [λ for x in X]
    ρVec = [ρ for x in X]

    S12 = 1 ./λVec
    S21 = -ρVec.*ω^2

    y1 = zeros(Float64, N) # displacement
    y2 = zeros(Float64, N) # traction

    # free surface with normalized displacement
    y1[1] = 1.0

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


"""
Compute the compressive normal modes of an one dimension rode.

INPUTS
    ρ(x) {function} : density
    λ(x) {function} : elastic modulus
    ω1,ω2 {Float} : min and max frequencies
    L {FLoat} : bar length
    ε = 1e-2 {Float} : precision over traction at free surface
    debug = false : verbosity

OUTPUTS
    (ωn, un) : eigen frequencies and normal modes (interpolated at the same points)
"""
function computeNormalModes(ρ::Float64, λ::Float64, L::Float64, ω1::Float64, ω2::Float64; ε = 1e-2, debug = false)

    # finding the maximum velocity (needed to compute the integration step)

    αmax = sqrt(λ/ρ)
    
    # initial regular sampling to frame the zeros

    dω = (ω2 - ω1)/1000  
    Ω = collect(range(ω1, ω2, step = dω))

    Ωbounds = []

    debug && println("Searching for eigen frequencies bounds.")

    T0a = computeMode(ρ, λ, ω1, αmax, L)[2][end]

    for (i,ω) in enumerate(Ω[2:end])

        T0b = computeMode(ρ, λ, ω, αmax, L)[2][end]

        if T0b * T0a < 0
            push!(Ωbounds, (Ω[i], ω))
        end

        T0a = T0b
    end

    Nfp = length(Ωbounds)
    debug && println("At least $Nfp eigen frequencies to find.")

    # finding all eigenfrequencies by dichotomy

    eigenFreq = zeros(Float64, Nfp)

    for (ifp, (ωa, ωb)) in zip(1:Nfp, Ωbounds)

        debug && println(@sprintf("\nSearching normal mode between %.4g and  %.4g", ωa, ωb))

        T0a = computeMode(ρ, λ, ωa, αmax, L, debug = debug)[2][end]
        T0b = computeMode(ρ, λ, ωb, αmax, L)[2][end]

        ωc = (ωa + ωb)/2
        T0c = computeMode(ρ, λ, ωc, αmax, L)[2][end]

        nit = 0

        # on cherche un zéro de T(omega)
        while abs(T0c) > ε

            nit += 1

            ωc = (ωa + ωb)/2
            T0c = computeMode(ρ, λ, ωc, αmax, L)[2][end]

            if T0c*T0a < 0
                ωb = ωc
                T0b = T0c
            else
                ωa = ωc
                T0a = T0c
            end
        end

        eigenFreq[ifp] = ωc

        debug && println(@sprintf("Find a normal mode at ω = %.4g, %d iterations.", ωc, nit))

    end

    # computing the normal modes and interpolating them on the same points

    debug && println("\nComputing and interpolating the eigenfunctions at the same points.")

    ωp_max = eigenFreq[end]
    Nx = Int(div(L,αmax/ωp_max/5))
    X = collect(range(0.0, L, length = Nx))

    eigenFunctions = zeros(Float64, Nfp, Nx)

    for (ifp,ωp) in zip(1:Nfp,eigenFreq)

        u,T = computeMode(ρ, λ, ωp, αmax, L)
        X_b = collect(range(0.0, L, length = length(u)))
        eigenFunctions[ifp,:] = LinearInterpolation(X_b, u).(X)
    
    end

    
    debug && println("\nEnd of calculation of the normal mode catalogue.\n $Nfp modes found.")
    
    return eigenFreq, eigenFunctions

end


function computeNormalModes_bm(ρ::Float64, λ::Float64, L::Float64, ω1::Float64, ω2::Float64, to::TimerOutput; ε = 1e-2, debug = false)
    @timeit to "computing normal modes" begin
        computeNormalModes(ρ, λ, L, ω1, ω2, ε=ε, debug=debug)
    end
end

meanFunction(f, a::Float64, b::Float64; N = 100) = sum(f.(range(a, b, length = N)))/N

"""
COmpute the propagation of an elastic wave in the bar by normal mode summation.

INPUTS
    un : normal modes catalogue
    ωn : associated frequencies
    L : length of the bar
    xs : source position
    tf : total duration of the simulation
    Nt : number of time steps
    ρ : density
    g : source function
    debug = false : verbosity

OUTPUTS
    u : displacement over (t,x)
"""
function NMS(un::Array{Float64, 2}, ωn::Array{Float64, 1}, L::Float64, xs::Float64, tf::Float64, ρ::Float64, g::Array{Float64, 1} ; debug = false)

    u0 = 1/(ρ*L)

    Nfp, Nx = size(un)

    debug && println("Dimensions données d'entrées : $Nfp modes évalués sur $Nx points.")

    X   = collect(range(0.0, L, length = Nx))
    ixs = findfirst(X .>= xs)

    debug && println("Source en indice $ixs")

    # calcul pour une source en dirac

    Nt = length(g)
    time = collect(range(0.0, tf, length=Nt))
    ud(ix,t) = t*u0 + sum([sin(ωn[n]*t)/ωn[n]*un[n,ixs]*un[n,ix] for n=1:Nfp])

    ud_eval = [ud(i,t) for i in 1:Nx, t in time]

    # convolution avec la source

    # on prend un point des modes sur 10
    n_sample = 3
    Nx2 = Int(div(Nx, n_sample))

    u_eval = zeros(Float64, Nt, Nx2)

    for ix in 1:Nx2, it in 1:Nt
        u_eval[it,ix] = sum([ud_eval[ix*n_sample,jt]*g[it-jt] for jt=1:it-1])
    end

    return u_eval
end

function NMS_bm(un::Array{Float64, 2}, ωn::Array{Float64, 1}, L::Float64, xs::Float64, tf::Float64, ρ::Float64, g::Array{Float64, 1} , to::TimerOutput; debug = false)
    @timeit to "normal mode summation" begin
        NMS(un, ωn, L, xs, tf, ρ, g, debug=debug)
    end
end

# PLOTTING FUNCTIONS

"""
Plot NMS results at iteration it
"""
function plotNMSresults(u::Array{Float64, 2}, it::Int)
    plt.plot(u[it, :])
    plt.show()
end

"""
Plot NMS results as a record section.
"""
function plotNMSresults(u::Array{Float64, 2})

    time = collect(1:size(u)[1])
    plt.figure(figsize = (8,4))
    for i in 1:size(u)[2]
        plt.plot(u[:,i]./maximum(u[:,i])*2 .+i, time, "k", lw = 0.5)
    end
    plt.show()
end

"""
Plot the source function.
"""
function plotSource(g::Array{Float64, 1}, tf::Float64 ; Nt = 1000)
    time = collect(range(0.0, tf, length = Nt))
    plt.plot(time, g.(time))
    plt.xlabel("Time")
    plt.title("Source function")
    plt.show()
end

"""
Plot all the eigenfunctions (n=-1) or only one (n>=1).
"""
function plotModes(un::Array{Float64, 2}, L::Float64; n = -1)
 
    fig, ax = plt.subplots(tight_layout = true, figsize = (8,4))

    X = collect(range(0.0, L, length=size(un)[2]))
    if n != -1
        u = un[n,:]
        ax.plot(X, u)    
    else
        for n = 1:size(un)[1]
            ax.plot(X, un[n,:])
        end
    end
    plt.show()  
end

"""
Plot all the eigenfrequencies of the catalogue.
"""
function plotFrequencies(eigenFreq::Array{Float64, 1})

    fig, ax = plt.subplots(tight_layout = true)

    ax.plot(eigenFreq, ".")

    ax.set_xlabel("Overtone index \$n\$")
    ax.set_ylabel("Frequency")
    ax.grid()
    plt.show()
end

"""
Plot NMS results as an animation.
"""
function plotField(U::Array{Float64, 2}, tf::Float64)

    Nt = size(U)[1]
    dt = tf/Nt
 
    fig, ax = plt.subplots()
    
    umax = maximum(U)
    ax.set_ylim([-umax,umax])

    i = 1
    line, = ax.plot(U[i,:])
    ttl = ax.text(0.1, 0.75, "t = 0s", transform=ax.transAxes)

    ax.grid()
    
    i_update = 3
        
    function updatefig(i) 
        line.set_ydata(U[1+i*i_update,:])
        ttl.set_text(@sprintf("t = %.2es", i*i_update*dt))
        return (line,ttl)
    end

    anim = mpl.animation.FuncAnimation(fig, updatefig, interval=30, frames=div(Nt,i_update))

    anim.event_source.start()

    plt.show()

end


# SOURCE FUNCTIONS

"""
Ricker source function (second derivative of a gaussian)
    t0 : central time
    ω0 : central Frequency
"""
function source_ricker(t::Float64, t0::Float64, ω0::Float64)
    return exp(-((t-t0)*ω0)^2)*(4*(t-t0)^2*ω0^4 - 2*ω0^2)
end

"""
Heaviside source function (box spectrum smoothed at the edge with cosines)
    t0 : central time of the source function
    ωs =  (ω1, ω2, ω3, ω4) : the 4 frequencies describing the sepctrum
"""
# function source_heaviside(t0, ωs)
#     throw(error("Heaviside source function not yet implemented."))
# end


end # module


if abspath(PROGRAM_FILE) == @__FILE__

    using .NMS1d

    using TimerOutputs

    to = TimerOutput()

    L = 1.0

    ρ = 1.0
    λ = 1.0

    ω1 = 1e-5
    ω2 = 500.0

    # ωn, un = computeNormalModes(ρ, λ, L, ω1, ω2, debug = true)
    ωn, un = computeNormalModes_bm(ρ, λ, L, ω1, ω2, to)

    # plotModes(un, L)
    # plotFrequencies(ωn)

    # definition source

    tf = 1.0
    Nt = 1000
    t0 = 0.2
    ω0 = 30.0
    xs = 0.25

    time = collect(range(0.0, tf, length = Nt))
    source = source_ricker.(time, t0, ω0)

    # u = NMS(un, ωn, L, xs, tf, ρ, source, debug = true)
    u = NMS_bm(un, ωn, L, xs, tf, ρ, source, to)

    plotSource(g, tf)
    plotField(u, tf)



end