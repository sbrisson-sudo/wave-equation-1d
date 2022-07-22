
import Printf:@sprintf

import FFTW

import PyPlot
const plt = PyPlot
import PyCall: pyimport
const mpl = pyimport("matplotlib")
mpl.style.use("myStyle")

using TimerOutputs

function plot_results(u_nms::Array{Float64, 2}, x_nms::Array{Float64, 1}, u_sem::Array{Float64, 2}, x_sem::Array{Float64, 1}, tf::Float64; show = true, norm = false)

    Nt = size(u_nms)[1]
    dt = tf/Nt
 
    fig, ax = plt.subplots()

    # normalisation (???)

    it_norm = Int(div(Nt,2))
    
    if norm

        umax_sem = maximum(u_sem[it_norm,:])
        u_sem = u_sem ./ umax_sem

        umax_nms = maximum(u_nms[it_norm,:])
        u_nms = u_nms ./ umax_nms

    end

    ax.set_ylim([-2, 2])

    i = 1
    line_nms, = ax.plot(x_nms, u_nms[i,:], label = "NMS")
    line_sem, = ax.plot(x_sem, u_sem[i,:], label = "SEM")
    ttl = ax.text(0.1, 0.75, "t = 0s", transform=ax.transAxes)

    ax.grid()
    ax.legend()
    
    i_update = 3
        
    function updatefig(i) 
        line_nms.set_ydata(u_nms[1+i*i_update,:])
        line_sem.set_ydata(u_sem[1+i*i_update,:])
        ttl.set_text(@sprintf("t = %.2es", i*i_update*dt))
        return (line_nms, line_sem, ttl)
    end

    anim = mpl.animation.FuncAnimation(fig, updatefig, interval=30, frames=div(Nt,i_update))

    !show && return anim
    
    anim.event_source.start()
    plt.show()

end


"""
plot a trace and its spectrum.
"""
function plot_spectrum(u_nms::Array{Float64, 1}, u_sem::Array{Float64, 1}, tf::Float64, t0, ω0; normalize = false)

    if normalize
        norm(v::Array{Float64, 1}) = v ./ maximum(v)
        u_sem = norm(u_sem)
        u_nms = norm(u_nms)
    end

    fig, (ax_t, ax_f) = plt.subplots(1, 2, tight_layout = true, figsize = (12,6))

    fig.suptitle(@sprintf("Source : t0 = %.1fs, ω0 = %dHz", t0, ω0))

    for (u,label) in zip([u_sem, u_nms], ["SEM","NMS"])

        Nt = length(u)
        sr = Nt / tf

        uf = FFTW.rfft(u)
        Nf = length(uf)

        ax_t.plot(range(0.0, tf, length = Nt), u, label = label)

        ax_f.plot(collect(range(0.0, sr/2, length = Nf)), abs.(uf), ".-", label = label)
    end

    # plot the diff 

    ax_t.plot(range(0.0, tf, length = Nt), u_sem.-u_nms, label = "SEM - NMS")


    ax_t.grid()
    ax_t.legend()
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Displacement (m)")
    ax_f.grid()
    ax_f.legend()
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Power (m.s)")

end


    


if abspath(PROGRAM_FILE) == @__FILE__

    # program to compare solving the wave equation with NMS or SEM

    include("nms1d.jl")
    using .NMS1d

    include("sem1d.jl")
    using .SEM1d


    to = TimerOutput()

    # parametres du milieu

    L = 1.0     # taille du milieu
    ρ = x -> 1.0     # densité
    λ = x -> 1.0     # module élastique

    # source et temps de résolution

    tf = 1.0
    Nt = 1000
    Δt = tf / Nt
    time = collect(range(0.0, tf, length = Nt))

    t0 = 0.2
    ω0 = 30.0
    xs = 0.25

    f_source(t) = source_ricker(t, t0, ω0)

    #------------
    # SEM
    #------------

    Ne = 100    # nombre d'éléments
    N = 5       # degré polynomial des éléments

    Xsem = getNodesPosition(L, Ne, N)
    sIdx_sem = findfirst(Xsem .>= xs)

    println("[SEM] Computing mass and stiffness matrices...")
    M,K,f_p = initiateSEM_bm(ρ, λ, L, Ne, N, sIdx_sem, to)
    println("[SEM] Solving in time...")
    u_sem, xs2 = runSEM_bm(M, K, f_p, L, f_source, xs, tf, Nt, Ne, N, to)


    #------------
    # NMS
    #------------

    ω1 = 1e-5   # fmin base de modes
    ω2 = 400.0  # fmax base de modes

    println("[NMS] Computing the normal modes catalogue...")
    ωn, un = computeNormalModes_bm(ρ(0.0), λ(0.0), L, ω1, ω2, to)
    println("[NMS] Performing the normal modes summation...")
    u_nms = NMS_bm(un, ωn, L, xs2, tf, ρ(0.0), f_source.(time), to)

    Xnms = collect(range(0.0, L, length = size(u_nms)[2]))

    #----------
    # PLOT COMPARATIF
    #----------

    println("Benchmarking results :")
    show(to)
    println()

    xr = 0.8
    ix_sem = findfirst(Xsem .> xr)
    ix_nms = findfirst(Xnms .> xr)

    plot_spectrum(u_nms[:,ix_nms], u_sem[:,ix_sem], tf, t0, ω0)

    show_anim = false

    ani = plot_results(u_nms, Xnms, u_sem, Xsem, tf, show = show_anim)


    if !show_anim
        file = "nmsVSsem.mp4"
        println("Saving results in ", file)
        writer = mpl.animation.FFMpegWriter(fps=15, bitrate=-1) 
        ani.save(file, writer=writer, dpi = 300)
    end
    
end
    