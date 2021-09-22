using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

function PS4()

url="https://raw.githubusercontent.com/OU-PhD-EConometrics/fall-2021/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df=CSV.read(HTTP.get(url).body, DataFrame)
X= [df.age df.white df.collgrad]
Z=hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y=df.occ_code

function mlogit(Beta, X, Z,y)
    
        K = size(X,2)+1
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigBeta = [reshape(Beta,K,J-1) zeros(K)]
    
        bigZ = zeros(N,J)
        for j=1:J
            bigZ[:,j] = Z[:,j]-Z[:,J]
        end
    
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            XZ=cat(X,bigZ[:,j],dims=2)
            num[:,j] = exp.(XZ*bigBeta[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end
    
    mlogit_hat_optim = optimize(b-> mlogit(b,X,Z, y), rand(7*(size(X,2)+1)), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    println(mlogit_hat_optim.minimizer)

2. #The coefficient is positive which is closer to our expecation.

3. 
using Distributions
include ("lgwt.jl)

#a

d=Normal(0,1) 

nodes,weights = lgwt(7,-4,4)

sum(weights.*pdf.(d,nodes))

sum(weights.*nodes.*pdf.(d,nodes))

#b

d=Normal(0,2)
nstd = 5*sqrt(2)
nodes, weights = lgwt (7, -nstd, nstd)

sum(weights.*nodes.*nodes,*pdf.(d,nodes))

nodes, weights = lgwt (10, -nstd, 5*nstd)

sum(weights.*nodes.*nodes,*pdf.(d,nodes))

#the 10 grid setup provides a closer approximation

#c

NumObs = 10^6

x= rand(Normal(0,2),NumObs)
x_sq= x.*X

function f(y)
1/(2*sqrt(2*pi)) * exp(-.5*(y/2)^2)
end

w=(nstd)/NumObs
func1=map(y->f(y),x) .*x_sq
MC1= w * sum(func1)
println

#close to 4

func2= map(y->f(y),x) .* X
MC1= w * sum(func2)
println

#close to 0

func3= map(y->f(y),x)
MC3= w * sum(func3)
println

# Is not close to one 

#d

w .* (x_sq .*pdf.(Normal(0,2),x))

w .* (x.* pdf.(Normal(0,2),x))

w .* pdf(Normal(0,2),x)

10^6 is the better approximation

#d Not sure there was really a question there

#4 ??????????????????????????????



#5 ??????????????????????????????

#6 PS(4)