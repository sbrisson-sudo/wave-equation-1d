module SEM1d

import Printf: @sprintf
import DelimitedFiles: readdlm
using PyPlot
const plt = PyPlot
import PyCall: pyimport

const mpl = pyimport("matplotlib")
const anim = pyimport("matplotlib.animation")
# mpl.style.use("myStyle")

using TimerOutputs

export initiateSEM, runSEM, getNodesPosition
export initiateSEM_bm, runSEM_bm
export plotField


"""
ξi, ωi, dhij = getGLL(N, dir="gll_quad/") 
"""
function getGLL(N::Int, dir="gll_quad/")
    datafile = @sprintf("%sgll_%02d.tab",dir, N)
    data = readdlm(datafile, Float64)
    return data[1,:], data[2,:], data[3:end,:] 
end

# passage d'indexage (e,i) élément-point à indexage local (points dédoublés) et global (pas de points dédoublés)
localIdx(e, i, N) = (e-1)*(N+1) + i
globalIdx(e, i, N) = (e-1)*N + i
iglobalIdx(α, N) = α==1 ? (1,1) : (div(α-1,N)+1, α-div(α-1,N)*N)

function getNodesPosition(L::Float64, Ne::Int, N::Int)

    ξi, _,_ = getGLL(N+1)
    le = L/Ne
    nodes = fill(0.0, Ne*N+1)
    for α=1:Ne*N+1
        e,i = iglobalIdx(α, N)
        nodes[α] = le*(e-1 + (ξi[i]+1)/2)
    end
    return nodes
end


"""
Compute the SEM mass and stiffness matrices.
INPUTS
    - ρ {function} : density
    - λ {function} : elastic modulus
    - L {float} : length of the bar
    - Ne {integer} : number of elements
    - N {integer} : order of polynomials to use
    - is {integer} : position de la source (global numbering)


"""
function initiateSEM(ρ, λ, L, Ne, N, is_g)

    ξi, ωi, dhij = getGLL(N+1) # GLL points + lagrange interpolants derivative at theses points

    elmtsPosition = collect(range(1, L, length = Ne))

    ρElmts = ρ.(elmtsPosition)
    λElmts = λ.(elmtsPosition)

    le = L/Ne
    Je = le/2.0 # jacobian

    # CONSTRUCTION DES MATRICES LOCALES

    ML = fill(0.0, (Ne*(N+1), Ne*(N+1)))
    KL = fill(0.0, (Ne*(N+1), Ne*(N+1)))

    # MATRICE DE MASSE

    for e=1:Ne, i=1:N+1
        α = localIdx(e, i, N)
        ML[α,α] = ωi[i] * ρElmts[e] * Je
    end

    # STIFFNESS MATRIX

    for e=1:Ne, i=1:N+1, j=1:N+1, k=1:N+1
        αi, αj = localIdx(e,i,N), localIdx(e,j,N)
        KL[αi, αj] += ωi[k] * λElmts[e] * dhij[j,k] * dhij[i,k] /Je
    end

    # ASSEMBLAGE EN MATRICES GLOBALES

    Q = fill(0.0, (Ne*N+1, Ne*(N+1)))

    for e=1:Ne, i=1:N+1
        α = localIdx(e,i,N)
        β = globalIdx(e,i,N)
        Q[β,α] = 1.0
    end

    M = Q*ML*Q'
    K = Q*KL*Q'

    # SOURCE

    es, is = iglobalIdx(is_g, N)
    f_prefactor = ωi[is] * Je

    return M,K,f_prefactor

end # function

function initiateSEM_bm(ρ, λ, L, Ne, N,is_g, to::TimerOutput)
    @timeit to "compute SEM matrices" begin 
        initiateSEM(ρ, λ, L, Ne, N, is_g)
    end # @timeit
end



"""
Run a SEM simulation.
INPUTS
    - M : mass MATRIX
    - K : stiffness matrix 
    - L
    - f {function} : source function
    - xs {float} : source position
    - T {float} : time duration
    - Δt {float} : time step
"""
function runSEM(M, K, f_prefactor, L, f, xs, tf, Nt, Ne, N)

    Δt = tf/Nt

    dof = size(M)[1]

    nodesPos = getNodesPosition(L, Ne, N)
    xsIdx = findfirst(nodesPos .> xs)
    xs_eff = nodesPos[xsIdx]

    iM = inv(M)

    # CALCUL DYNAMIQUE

    U = fill(0.0, (Nt,dof))
    F = fill(0.0, dof)

    for i=3:Nt
        F[xsIdx] = f(i*Δt)* f_prefactor
        U[i,:] = 2 .*U[i-1,:] .- U[i-2,:] + Δt^2 .* (iM*(F-K*U[i-1,:]))
    end

    return U, xs_eff
end # function

function runSEM_bm(M, K, f_p, L, f, xs, tf, Nt, Ne, N, to::TimerOutput)
    @timeit to "Run SEM Newmark scheme" begin
        runSEM(M, K, f_p, L, f, xs, tf, Nt, Ne, N)
    end
end

"""
Plot SEM results as an animation.
"""
function plotField(X, U)

    Nt = size(U)[1]
 
    fig, ax = plt.subplots()
    
    umax = maximum(U)
    ax.set_ylim([-umax,umax])

    i = 1
    line, = ax.plot(X, U[i,:])
    ttl = ax.text(0.1, 0.75, "it = 0", transform=ax.transAxes)

    ax.grid()
    
    i_update = 3
        
    function updatefig(i) 
        line.set_ydata(U[1+i*i_update,:])
        ttl.set_text("it = $(i*i_update)")
        return (line,ttl)
    end

    myAnim = anim.FuncAnimation(fig, updatefig, interval=30, frames=div(Nt,i_update))

    myAnim.event_source.start()

    plt.show()

end


end

if abspath(PROGRAM_FILE) == @__FILE__

    using .SEM1d

    using TimerOutputs
    to = TimerOutput()

    L = 1.0
    ρ = x -> 1.0
    λ = x -> 1.0

    Ne = 100
    N = 5

    # definition source

    tf = 1.0
    Nt = 1000
    t0 = 0.1
    ω0 = 30
    xs = 0.25

    X = getNodesPosition(L, Ne, N)

    xsIdx = findfirst(X .> xs)

    println(">> SEM inititiation...")
    # M,K = initiateSEM(ρ, λ, L, Ne, N)
    M,K,f_p = initiateSEM_bm(ρ, λ, L, Ne, N, xsIdx, to)

    f(t) = exp(-((t-t0)*ω0)^2)*(4*(t-t0)^2*ω0^4 - 2*ω0^2)

    println(">> SEM simulation...")
    # u,_ = runSEM(M, K, L, f, xs, tf, Δt, Ne, N)
    u,_ = runSEM_bm(M, K, f_p, L, f, xs, tf, Nt, Ne, N, to)

    println(to)

    println(">> Plotting...")
    plotField(X, u)


end

