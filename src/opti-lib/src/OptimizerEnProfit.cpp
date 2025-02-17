#include "OptimizerEnProfit.h"
#include "IOptimizable.h"
#include "IPeriod.h"
#include "ConfigGlob.h"
#include "ConfigMan.h"
#include "ConfigOpti.h"
#include "ConfigSol.h"

#include "TradeUtil.h"
#include "OptiVarVec.h"
#include "OptiType.h"
//#include "OptiGoalFactory.h"
#include "OptiEnProfitSubject.h"
#include "OptiEnProfitDataModel.h"
#include "OptiEnProfitResults.h"
#include "GnuplotIOSWrap.h"
#include "SolUtil.h"
#include "TimeUtil.h"

#include <Math/GeneralMath.hpp>
#include <Math/RandomMath.hpp>
#include <Util/ProgressMonit.hpp>
#include <Util/ProgressMonitHigh.hpp>
#include <Util/CharManipulations.hpp>
#include <Util/ToolsMixed.hpp>
#include <Util/Except.hpp>
#include <Util/CoutBuf.hpp>
#include <Util/StrColour.hpp>
#include <Statistical/Statistical.hpp>
#include <Statistical/Distrib.hpp>
#include <Template/CorradePointer.h>
#include <Visual/AsciiPlot.hpp>

#include "OptiEnProfitSubject.h" /// TODO: Remove

#include <STD/VectorCpp.hpp>
#include <STD/Set.hpp>
#include <STD/String.hpp>

#include <limits>

using namespace std;
using namespace EnjoLib;

const int OptimizerEnProfit::HOURS_IN_DAY = 24;
const double OptimizerEnProfit::GOAL_INITIAL = std::numeric_limits<double>::min();

const OptimizerEnProfit::BigInt OptimizerEnProfit::MAX_NUM_COMBINATIONS = std::numeric_limits<BigInt>::max(); //1e7;
//const OptimizerEnProfit::BigInt OptimizerEnProfit::MAX_NUM_COMBINATIONS = 3e6;
const double OptimizerEnProfit::MAX_FAILED_COMBINATIONS = 0.60;
const double OptimizerEnProfit::MIN_POS_2_NEG_CHANGE_RATIO = 0.01;

OptimizerEnProfit::OptimizerEnProfit(const OptiEnProfitDataModel & dataModel)
    : m_dataModel(dataModel)
{
}
OptimizerEnProfit::~OptimizerEnProfit() {}

bool OptimizerEnProfit::IsUseHash() const
{
    return true;
    //return false;

    const int MIN_HOURS_HASHMAP = 2 * HOURS_IN_DAY;
    const int hours = m_dataModel.GetHorizonHours();
    return hours > MIN_HOURS_HASHMAP;
}

/// TODO: The basic multi-dim-iter interaction should go to upper library
void OptimizerEnProfit::operator()()
{
    ELO
    float goal = 0;
    //const bool randomSearch = false;
    const bool randomSearch = true;
    if (randomSearch)
    {
        RandomSearch();
    }
    else
    {
        //multiDimIter.StartIteration(data, *this);
    }
}

EnjoLib::Str OptimizerEnProfit::GetT() const
{
    return SolUtil().GetT();
}

void OptimizerEnProfit::RandomSearch()
{
    const ConfigSol & conf = m_dataModel.GetConf();
    const int horizonHours = m_dataModel.GetHorizonHours();
    const EnjoLib::Array<Computer> & comps = m_dataModel.GetComputers();
    const int numComputers = comps.size();

    const GMat gmat;

    // A heuristic to get the number of possible combinations.
    // TODO: Should check the variance changes
    const BigInt maxIter = gmat.Pow(gmat.Pow(horizonHours, 3), gmat.Pow(numComputers, 1/2.0));

    {LOGL << GetT() << "Random search of " << maxIter << " solutions; "
    << horizonHours << "h, rigs = " << numComputers << Nl;}

    const RandomMath rmath;
    rmath.RandSeed(conf.RANDOM_SEED);
    const VecD binaryZero(horizonHours);
    const std::string hashStrZero(horizonHours * (numComputers > 0 ? numComputers : 1), '0');
    std::string hashStr = hashStrZero;
    Matrix binaryMat;
    VecT<int> minHoursTogetherHalfVec;
    for (const Computer & comp : comps)
    {
        binaryMat.Add(binaryZero);
        const int minHoursTogetherHalf = GMat().round(comp.minRunHours/2.0);
        minHoursTogetherHalfVec.push_back(minHoursTogetherHalf);
    }
    if (numComputers == 0)
    {
        binaryMat.Add(binaryZero);
        minHoursTogetherHalfVec.push_back(1);
    }
    const Matrix binaryMatZero = binaryMat;
    m_binarBest = binaryMat;

    bool foundFirstSolution = false;
    const bool useHash = IsUseHash();
    const BigInt maxCombisFailed = maxIter * MAX_FAILED_COMBINATIONS;
    const short bit = 1;
    const char bitC = '1';
    std::set<std::string> usedCombinations;
    int alreadyCombined = 0;
    //const Distrib distr;
    const SolUtil sut;
    std::vector<Sol0Penality> solutions0Penality;
    const bool animateProgressBar = m_dataModel.IsAnimateProgressBar();
    ProgressMonitHigh progressMonitor(13);
    bool needNewLine = false;
    BigInt iter = 0;
    for (BigInt i = 0; i < maxIter; ++i)
    {
        if (animateProgressBar)
        {
            if (i % 100000 == 0)
            {
                //const Str & progressStr = "Solutions";
                //const Str & progressStr = OptiEnProfitResults().PrintOptiPenality(m_goals, 10);
                const Str & progressStr = OptiEnProfitResults().PrintOptiSingle(m_hashesProgress, 10);
                progressMonitor.PrintProgressBarTime(i, maxIter, progressStr);
                //if (i > 0)
                {
                //const DistribData & data = distr.GetDistrib(m_goals, 20); const Str & dstr = distr.PlotLine(data, true, true, true);
                //progressMonitor.PrintProgressBarTime(i, maxIter, dstr);
                }
                needNewLine = true;
            }
        }

        const int icompRandom = gmat.round(rmath.Rand(0, numComputers-1));
        const int icomp = icompRandom >= 0 ? icompRandom : 0;
        //for (int icomp = 0; icomp < numComputers; ++icomp)
        {
            //LOGL << "icomp = " << icomp << Nl;
            VecD & binary = binaryMat.at(icomp);
            const int minHoursTogetherHalf = minHoursTogetherHalfVec.at(icomp);
            const int compIdxMul = 1 + icomp;
            const int index = gmat.round(rmath.Rand(0, horizonHours-1));
            //if (bit == 1)
            {
                for (int j = index - minHoursTogetherHalf; j <= index + minHoursTogetherHalf; ++j)
                {
                    if (j < 0 || j >= horizonHours)
                    {
                        continue;
                    }
                    binary[j] = bit; /// Each computer gets its own binary.
                    hashStr.at(j * compIdxMul) = bitC; /// TODO: j * (1 + computerIDX)
                }
            }
            int sum = 0;
            for (int l = 0; l < horizonHours; ++l)
            {
                sum += binary[l];
            }
            if (sum == horizonHours)
            {
                binary = binaryZero;
                for (int j = horizonHours * icomp; j < horizonHours * (icomp + 1); ++j)
                {
                    hashStr.at(j) = '0'; /// TODO: Something is wrong here
                }
                //ELO
                //LOG << "Hash pre: " << hashStr << Nl;
                //hashStr = hashStrZero;
                //bit = 1;
                //LOG << "\nHash post: " << hashStr << Nl;
            }
        }
        bool found = false;
        if (useHash)
        {
            found = not usedCombinations.insert(hashStr).second;
            //found = usedCombinations.count(hashStr);
        }
        if (found)
        {
            ++alreadyCombined;
            ++m_numFailed;
        }
        else
        {
            if (Consume2(binaryMat, needNewLine))
            {
                SOL_LOG(GetT() + "Consume success: " + binaryMat.Print());
                //LOGL << "Consume success: " << binaryMat.Print() << '\n';
                m_numFailed = 0;
                m_binarBest = binaryMat;
                m_uniqueSolutionsPrev = m_uniqueSolutions;
                m_uniqueSolutions = usedCombinations.size();
                const double sumBest = sut.SumMat(binaryMat);
                foundFirstSolution = sumBest != 0;
                needNewLine = false;

                if (m_penality == 0)
                {
                    Sol0Penality soldat;
                    soldat.sol = m_currSolution;
                    soldat.dat = binaryMat;
                    solutions0Penality.push_back(soldat);
                }
            }
            else
            {
                //SOL_LOG("Failed: " + binaryMat.Print());
                ++m_numFailed;
            }
            RecalcComputationCosts();
        }
        const bool changeLargeEnough = m_relPos2Neg == 0 || m_relPos2Neg > MIN_POS_2_NEG_CHANGE_RATIO;
        const bool exceededNumFailed = m_numFailed >= maxCombisFailed;
        //if (exceededNumFailed && not changeLargeEnough)
        if (exceededNumFailed && foundFirstSolution)
        {
            LOGL << "Early stop after " << m_numFailed << " last failed attempts "
                 // << and last change of " << m_relPos2Neg << " < " << MIN_POS_2_NEG_CHANGE_RATIO << Nl
                 << Nl
                 << "Repeated combinations = " << alreadyCombined << " of " << maxCombisFailed << ": " << GMat().round(alreadyCombined/double(maxCombisFailed) * 100) << "%" << Nl
                 << "Unique   combinations = " << usedCombinations.size() << " of " << maxCombisFailed << ": " << GMat().round(usedCombinations.size()/double(maxCombisFailed) * 100) << "%" << Nl;
            break;
        }
        ++iter;
    }
    {LOGL << Nl << GetT() << "Finished after " << iter << " iterations." << Nl;}
    const Str notFoundSolutionWarn = StrColour::GenWarn("Couldn't find a solution!\n"
                                                        "The usual remedy is to increase the number of batteries, "
                                                        "or reduce the load in 'habits' configuration file.\n"
                                                        "If the batteries are overcharged already, "
                                                        "the best thing to do is to start your machines now."
                                                        );
    if (not foundFirstSolution)
    {
        // TODO: Unit test it.
        //Assertions::Throw("Couldn't find a solution!", "OptimizerEnProfit::RandomSearch");
        //LOGL << Nl << notFoundSolutionWarn << Nl;
    }

    if (solutions0Penality.empty())
    {
        ELO
        LOG << OptiEnProfitResults().PrintOptiProgression(m_goals, m_hashesProgress, horizonHours);
        LOG << OptiEnProfitResults().PrintSolution(m_dataModel, m_binarBest);
    }
    else
    {
        ELO
        //LOG << StrColour::GenWarn("Got solutions: ") << solutions0Penality.size() << Nl;
        LOG << OptiEnProfitResults().PrintOptiProgression(m_goals, m_hashesProgress, horizonHours);
        LOG << OptiEnProfitResults().PrintMultipleSolutions(m_dataModel, solutions0Penality, conf.NUM_SOLUTIONS);
    }

    if (not foundFirstSolution)
    {
        LOGL << Nl << notFoundSolutionWarn << Nl;
    }
}

bool OptimizerEnProfit::Consume2(const EnjoLib::Matrix & dataMat, bool needNewline)
{
    OptiSubjectEnProfit osub(m_dataModel);
    const Solution & goal = osub.GetVerbose(dataMat, false);
    if (not goal.acceptable)
    {
        return false;
    }
    //LOGL << "goal = " << goal << Nl;
    if (goal.penality < m_goal || m_goal == GOAL_INITIAL || goal.penality == 0)
    {
        m_goals.Add(goal.penality);
        m_hashesProgress.Add(goal.hashes);
        const double relChangePositive = GMat().RelativeChange(goal.penality, m_goal);
        m_relChangePositive = relChangePositive;
        ELO
        RecalcComputationCosts();
        if (goal.penality < m_goal)
        {
            if (not m_dataModel.GetConf().NO_NEW_SOLUTIONS)
            {
                if (needNewline)
                {
                    LOG << Nl; // Need an extra space to clear the progress bar
                }
                LOG << GetT() << "New score: Penalty = " << -goal.penality << "\t, hashes = +" << goal.hashes << Nl;
                /*
                << " ->\t"
                << GMat().round(relChangePositive   * 100) << "%" << " costing: "
                << GMat().round(m_relChangeNegative * 100) << "%" << ", pos2neg: "
                << GMat().round(m_relPos2Neg        * 100) << "%" << Nl;
        //        << GMat().round(relNeg2Pos * 100) << "%" << Nl;
                */
            }

        }


        //osub.GetVerbose(dataMat, true);
        osub.GetVerbose(dataMat, false); /// TODO: Why would we recalculate it in a non-verbose mode? To store the final result?
        m_goal = goal.penality;
        m_penality = osub.GetPenality();

        if (goal.penality == 0)
        {
            m_currSolution = goal;
            //LOG << GetT() << "0 = Penality!" << Nl;
        }

        m_hashes = goal.hashes;

        return true;
    }
    else
    {
        return false;
    }
}


void OptimizerEnProfit::AddSpace(const EnjoLib::VecD & data)
{
    m_data.push_back(data);
}

void OptimizerEnProfit::Consume(const EnjoLib::VecD & data)
{

}

static double GMatRatio(double val, double valRef)
{
    if (valRef == 0)
    {
        return 0;
    }
    return val / valRef;
}


void OptimizerEnProfit::RecalcComputationCosts()
{
    m_relChangeNegative = GMat().RelativeChange(m_uniqueSolutions, m_uniqueSolutionsPrev);
    m_relPos2Neg = GMatRatio(m_relChangePositive, m_relChangeNegative);
}
