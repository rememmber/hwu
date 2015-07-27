{-# LANGUAGE ForeignFunctionInterface #-}
module Safe where

import Foreign
import Foreign.C.Types
import Data.Vector.Storable
import qualified Data.Vector.Storable as V
--import Network
import AI.HNN.Recurrent.Network
import AI.HNN.FF.Network
import Numeric.LinearAlgebra
import Data.List.Split
import Data.Matrix
import Data.Vector
import Data.List.Split

foreign export ccall process_network_input :: Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr (Ptr Double) -> IO ()
foreign export ccall train :: Ptr CInt -> IO ()

get1st (a,_) = a
get2nd (_,a) = a

concatr :: Double -> [[Double]] -> [[Double]]
concatr x ys = Prelude.map (x:) (rev ys)

foo xs = [(Prelude.length xs-1), (Prelude.length xs -2)..0]
rev xs = [xs !! k| k <- foo xs]

train :: Ptr CInt -> IO ()
train n = do 
    net <- (AI.HNN.FF.Network.createNetwork 6 [12] 4) :: IO (AI.HNN.FF.Network.Network Double)
    let samples = [ Numeric.LinearAlgebra.fromList [0.511346, 0.50056, 0.503204, 0.50056, 0.503005, 0.501285] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.5116, 0.500633, 0.503126, 0.500633, 0.503209, 0.501458] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.512747, 0.5003, 0.50059, 0.501063, 0.50883, 0.502629] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.51887, 0.501885, 0.500475, 0.502419, 0.512605, 0.507112] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.529106, 0.5, 0.5, 0.5, 0.511259, 0.5] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.510466, 0.5, 0.5, 0.5, 0.527142, 0.500091] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.527132, 0.5, 0.5, 0.500583, 0.52647, 0.501931] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.517094, 0.5, 0.501667, 0.500772, 0.508567, 0.502664] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.514571, 0.5, 0.501337, 0.500807, 0.508033, 0.502601] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.5091, 0.5, 0.5, 0.500188, 0.514712, 0.501018] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.536659, 0.5, 0.5, 0.500845, 0.527941, 0.502959] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.526258, 0.5, 0.500784, 0.500801, 0.520145, 0.502774] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.514195, 0.5, 0.5, 0.5, 0.536633, 0.505165] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.520485, 0.5, 0.5, 0.501013, 0.509218, 0.504634] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.510526, 0.500565, 0.502734, 0.500565, 0.503175, 0.501283] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.52439, 0.5, 0.502139, 0.500773, 0.516359, 0.502744] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.511329, 0.500619, 0.503221, 0.500619, 0.502814, 0.501394] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.52223, 0.5, 0.5, 0.5, 0.527427, 0.500543] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.513669, 0.500447, 0.500447, 0.500447, 0.500216, 0.500447] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.503591, 0.5, 0.5, 0.5, 0.504165, 0.500181] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.555134, 0.5, 0.5, 0.5, 0.511437, 0.5] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.534741, 0.5, 0.5, 0.500844, 0.526305, 0.502835] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.555134, 0.5, 0.5, 0.5, 0.511437, 0.5] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.502678, 0.5, 0.5, 0.5, 0.518415, 0.5] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.513308, 0.500496, 0.500496, 0.500496, 0.506195, 0.500496] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.514502, 0.500299, 0.500618, 0.501468, 0.50962, 0.503257] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.511177, 0.500282, 0.500451, 0.500695, 0.50875, 0.502224] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.528794, 0.5, 0.500861, 0.500762, 0.522081, 0.502641] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.518215, 0.5, 0.502088, 0.500337, 0.509882, 0.503084] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.512748, 0.500283, 0.500545, 0.501297, 0.509405, 0.503077] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.525938, 0.5, 0.502231, 0.500772, 0.517394, 0.502592] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.542486, 0.500297, 0.500528, 0.502237, 0.524597, 0.505443] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.519072, 0.5, 0.501269, 0.500787, 0.511958, 0.502357] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.518771, 0.5, 0.501449, 0.500207, 0.512877, 0.501656] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.52841, 0.5, 0.500786, 0.50035, 0.515106, 0.503768] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.551207, 0.500291, 0.50029, 0.500291, 0.509091, 0.500291] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                    , Numeric.LinearAlgebra.fromList [0.510791, 0.500234, 0.500442, 0.501093, 0.50877, 0.502399] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.511149, 0.500267, 0.500609, 0.501428, 0.508408, 0.503062] --> Numeric.LinearAlgebra.fromList [0, 0, 1, 0]
                    , Numeric.LinearAlgebra.fromList [0.523794, 0.5, 0.50164, 0.500765, 0.516305, 0.502528] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0]
                    , Numeric.LinearAlgebra.fromList [0.552369, 0.500276, 0.500275, 0.500276, 0.50841, 0.500889] --> Numeric.LinearAlgebra.fromList [0, 0, 0, 1]
                  ]
    let smartNet = trainNTimes 1000 0.8 AI.HNN.FF.Network.sigmoid sigmoid' net samples
    --print smartNet
    saveNetwork "smartNet.nn" smartNet

feed2 :: Int -> [Double] -> [Double] -> [Double -> Double] -> IO (Data.Vector.Storable.Vector Double)
feed2 nodes_number weights inputs_ functions = do
    --let weights_layers = Prelude.splitAt 72 weights
    --let weights_list = Prelude.concat (concatr 0.0 (splitEvery 6 (get1st weights_layers)))
    --let input_layer__ = (12><7) weights_list
    --let hidden_layer__ = (4><12) (get2nd weights_layers)
    --let vector_ = Data.Vector.fromList [input_layer__, hidden_layer__] --also had 'output_layer__' before
    --let n = AI.HNN.FF.Network.fromWeightMatrices vector_ :: AI.HNN.FF.Network.Network Double
    --let inputs__ = Numeric.LinearAlgebra.fromList inputs_
    --return (output n AI.HNN.FF.Network.sigmoid inputs__)
    n <- AI.HNN.FF.Network.loadNetwork "smartNet.nn" :: IO (AI.HNN.FF.Network.Network Double)
    --print n
    let inputs__ = Numeric.LinearAlgebra.fromList inputs_
    return (output n AI.HNN.FF.Network.sigmoid inputs__)


feed :: Int -> [Double] -> [[Double]] -> [Double -> Double] -> IO (Data.Vector.Storable.Vector Double)
feed nodes_number weights inputs_ functions = do
    let numNeurons = 442
        numInputs  = 441
        thresholds = Prelude.replicate numNeurons 0.0
        inputs     = inputs_
        adj        = weights
    n <- AI.HNN.Recurrent.Network.createNetwork numNeurons numInputs adj thresholds :: IO (AI.HNN.Recurrent.Network.Network Double)
    output <- evalNet n inputs AI.HNN.Recurrent.Network.sigmoid
    return output

peekInt :: Ptr CInt -> IO Int
peekInt = fmap fromIntegral . peek

build_function_list :: [Double] -> [Double -> Double]
build_function_list (x:xs) | x == 0.0 = AI.HNN.Recurrent.Network.sigmoid : build_function_list xs
                           | otherwise = AI.HNN.Recurrent.Network.sigmoid : build_function_list xs

process_network_input :: Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr (Ptr Double) -> IO ()
process_network_input atype n m nodes_number weights inputs functions result = do
    atype <- peekInt atype
    n <- peekInt n
    m <- peekInt m
    nodes_number <- peekInt nodes_number
    weights_ <- peekArray n weights
    inputs_ <- peekArray m inputs
    let inputs__ = inputs_:[]
    functions_ <- peekArray nodes_number functions
    let functions_list = build_function_list functions_
    res <- case (atype == 0) of
        True -> (feed (Prelude.length functions_) weights_ inputs__ functions_list) --used inputs__ before
        False -> (feed2 (Prelude.length functions_) weights_ inputs_ functions_list) --used inputs__ before
    let aList = (V.toList res)
    let b = (fromIntegral (Prelude.length aList)):aList
    ptr <- newArray b
    poke result $ ptr
