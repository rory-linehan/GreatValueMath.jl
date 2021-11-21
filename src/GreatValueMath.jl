module GreatValueMath

using Statistics
using DataFrames

# min-max feature scaling
function normalize(data::DataFrame, field::String)::DataFrame
    ma = maximum(data[!, field])
    mi = minimum(data[!, field])
    for (index, i) in enumerate(eachrow(data))
        data[index, field] = (i[field] - mi)/(ma - mi)
    end
    return data
end

function theil_sen_linear_regression(data::DataFrame, x::String, y::String)
    𝐧 = nrow(data)
    if 𝐧 > 1
      # independent variable vector
      xᵢ = data[!, x]
      # dependent variable vector
      yᵢ = data[!, y]
      # mean of dependent vector
      yₘ = (1/𝐧)*sum(yᵢ)
      # https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator
      m = median([(yᵢ[i+1]-yᵢ[i])/(xᵢ[i+1]-xᵢ[i]) for i ∈ 1:𝐧-1])
      𝑏 = median([(yᵢ[i]-m*xᵢ[i]) for i ∈ 1:𝐧])
      # https://en.wikipedia.org/wiki/Coefficient_of_determination
      # predicted vector
      fᵢ = [(m*x+𝑏) for x ∈ xᵢ]
      # residual vector
      𝖾ᵢ = [(yᵢ[i]-fᵢ[i]) for i ∈ 1:𝐧]
      SStot = sum([abs2(yᵢ[i]-yₘ) for i ∈ 1:𝐧])
      SSres = sum([abs2(𝖾ᵢ[i]) for i ∈ 1:𝐧])
      𝐑² = 1 - (SSres/SStot)
      return Dict("𝐑²" => 𝐑², "m" => m, "𝑏" => 𝑏)
    else
      return false
    end
end

end # module
