#include "OptiEnProfitResults.h"
#include "OptimizerEnProfit.h"

#include "OptiEnProfitDataModel.h"
#include "OptiEnProfitSubject.h"
#include "TimeUtil.h"
#include "SolUtil.h"
#include "ConfigSol.h"

#include <Ios/Osstream.hpp>
#include <Ios/Ofstream.hpp>
#include <Statistical/Distrib.hpp>
#include <Statistical/Assertions.hpp>
#include <Math/GeneralMath.hpp>
#include <Util/CoutBuf.hpp>
#include <Util/StrColour.hpp>
#include <Util/ToolsMixed.hpp>
#include <Util/CharManipulations.hpp>
#include <Visual/AsciiPlot.hpp>
#include <Visual/AsciiMisc.hpp>

#include <STD/Algorithm.hpp>

using namespace std;
using namespace EnjoLib;

const int OptiEnProfitResults::SSH_TIMEOUT_S = 60;

OptiEnProfitResults:: OptiEnProfitResults() {}
OptiEnProfitResults::~OptiEnProfitResults() {}

EnjoLib::Str OptiEnProfitResults::PrintMultipleSolutions(const OptiEnProfitDataModel & dataModel,
                                                         const std::vector<Sol0Penality> & solutions0Penality, int maxSolutions) const
{
    Osstream oss;
    std::vector<Sol0Penality> sols0Pen = solutions0Penality;

    std::sort   (sols0Pen.begin(), sols0Pen.end());
    std::reverse(sols0Pen.begin(), sols0Pen.end());
    std::vector<Sol0Penality> sols0PenSelected;
    for (int i = 0; i < sols0Pen.size() && i < maxSolutions; ++i)
    {
        const Sol0Penality & soldat = sols0Pen.at(i);
        sols0PenSelected.push_back(soldat);

        //PrintSolution(soldat.dat);
    }
    const Sol0Penality & soldatBest = sols0Pen.at(0);
    std::reverse(sols0PenSelected.begin(), sols0PenSelected.end());
    for (int i = 0; i < sols0PenSelected.size(); ++i)
    {
        ELO
        const Sol0Penality & soldat = sols0PenSelected.at(i);
        const double hashes = soldatBest.sol.hashes;
        //{
            const int len = 20;
            LOG << AsciiMisc().GenChars("-", len) << Nl;
            if (maxSolutions > 1)
            {
                LOG << "Solution " << i+1 << " of " << maxSolutions << Nl;
            }
            LOG << AsciiMisc().GenChars("-", len) << Nl;
        //}

        LOG << OptiEnProfitResults().PrintSolution(dataModel, soldat.dat, hashes);
    }
    //const Sol0Penality & soldatBest = sols0Pen.at(sols0Pen.size() - 1);
    //PrintSolution(soldatBest.dat);

    return oss.str();
}

/// TODO: UTest & refactor
EnjoLib::Str OptiEnProfitResults::PrintSolution(const OptiEnProfitDataModel & dataModel, const EnjoLib::Matrix & bestMat, double maxHashes) const
{
    Osstream ossLog;
    OptiSubjectEnProfit osub(dataModel);
    osub.GetVerbose(bestMat, true, maxHashes);
    const CharManipulations cman;
    const SolUtil sot;
    const ConfigSol & conf = dataModel.GetConf();
    for (int i = 0; i < bestMat.size(); ++i)
    {
        //GnuplotPlotTerminal1d(bestMat.at(i), "Best solution = " + CharManipulations().ToStr(m_goal), 1, 0.5);
    }
    //const Distrib distr;
    //const DistribData & distribDat = distr.GetDistrib(m_goals);
    //if (distribDat.IsValid())
    {
        //const Str & dstr = distr.PlotLine(distribDat, false, true, true);
        //ossLog << Nl << sot.GetT() <<  "Distribution of solutions:" << Nl<< dstr << Nl;
        //GnuplotPlotTerminal2d(distribDat.data, "Solution distribution", 1, 0.5);
    }

    ossLog << "\nComputer start schedule:\n";
    Osstream ossCmd;
    const int currHour = TimeUtil().GetCurrentHour();
    const int maxDayLimit = conf.DAYS_LIMIT_COMMANDS; /// TODO: Unstable, as it would require that at uses time AND day, not just time.
    if (maxDayLimit > 1)
    {
        Assertions::Throw("Not implemented max day limit > 1", "OptimizerEnProfit::PrintSolution");
    }
    const OptiEnProfitResults resPrinter;
    for (int i = 0; i < bestMat.size() && dataModel.GetComputers().size(); ++i)
    {
        const Computer & comp = dataModel.GetComputers().at(i);
        const VecD & best = bestMat.at(i);
        const OptiEnProfitResults::CommandsInfos & cmdInfo = resPrinter.PrintCommandsComp(conf, comp, best, currHour, maxDayLimit);
        ossLog << resPrinter.PrintScheduleCompGraph(comp, best);
        ossLog << cmdInfo.infos;
        ossCmd << cmdInfo.commands;
    }

    ossLog << "Commands:\n\n";
    ossLog << ossCmd.str();

    const Str fileCmds = conf.m_outDir + "/sol-cmds.sh";
    Ofstream ofs(fileCmds);
    ofs << ossCmd.str();

    ossLog << sot.GetT() << "Saved commands to:\n" << fileCmds << Nl;

    if (conf.NO_SCHEDULE)
    {
        return "";
    }

    return ossLog.str();
}

EnjoLib::Str OptiEnProfitResults::PrintOptiProgression(const EnjoLib::VecD & goals, const EnjoLib::VecD & hashesProgress, int horizonHours) const
{
    Osstream oss;
    oss << "Penalties/hashes progression: " << Nl;

    const int length = 24;
    //const int length = horizonHours; /// TODO: Uncovers a bug
    oss << PrintOptiPenality(goals, length) << Nl;
    oss << PrintOptiSingle(hashesProgress, length) << Nl;

    return oss.str();
}

EnjoLib::Str OptiEnProfitResults::PrintOptiPenality(const EnjoLib::VecD & penality, int horizonHours) const
{
    return PrintOptiSingle(penality.Abs(), horizonHours);
}

EnjoLib::Str OptiEnProfitResults::PrintOptiSingle(const EnjoLib::VecD & vec, int horizonHours) const
{
    Osstream oss;
    try
    {
        const double maxx = vec.Max();
        const double minn = vec.Min();
        oss << AsciiPlot::Build()
        (AsciiPlot::Pars::MAXIMUM, maxx)
        (AsciiPlot::Pars::MINIMUM, minn)
        (AsciiPlot::Pars::COMPRESS, horizonHours) /// TODO: uncovers a bug
        //(AsciiPlot::Pars::COMPRESS, 24) // "safe" option
        .Finalize().Plot(vec);
    }
    catch (const std::exception & exc)
    {
        oss << EnjoLib::StrColour::GenWarn("Printing failed!");
    }

    return oss.str();
}

EnjoLib::Str OptiEnProfitResults::PrintScheduleCompGraph(const Computer & comp, const VecD & best) const
{
    Osstream oss;
    oss << comp.name << Nl;
    const double maxx = 1;
    oss << AsciiPlot::Build()(AsciiPlot::Pars::MAXIMUM, maxx).Finalize().Plot(best) << Nl;
    return oss.str();
}

OptiEnProfitResults::CommandsInfos OptiEnProfitResults::PrintCommandsComp(const ConfigSol & conf, const Computer & comp, const VecD & best, int currHour, int maxDayCmdsLimit) const
{
    CommandsInfos ret;
    Osstream ossCmd, ossInfo;
    const CharManipulations cman;
    //oss << cman.Replace(best.Print(), " ", "") << Nl;
    ossCmd << "\"$(pwd)/util/jobs-remove-all.sh\"" << Nl;

    const Str cmdsSSHbare = "ssh -o ConnectTimeout=" + cman.ToStr(SSH_TIMEOUT_S) + " -n " + comp.hostname + " ";
    const Str cmdsSSH = "echo \"" + cmdsSSHbare + "'hostname";
    const Str cmdWOL = "wakeonlan " + comp.macAddr;
    //const Str cmdSuspendAt = "systemctl suspend\"           | at ";
    const Str cmdSysCtl = "; systemctl --no-wall ";
    const bool isPoweroff = conf.POWEROFF || comp.isPoweroff ;
    const Str cmdSuspendAt = cmdSysCtl + (isPoweroff ? "poweroff" : "suspend") + "'\" | at ";
    const Str cmdMinuteSuffix = ":00";

    bool onAtFirstHour = false;
    int lastHourOn = -1;
    int lastDayOn = -1;
    int firstDayOn = -1;
    //const int horizonHours = m_dataModel.GetHorizonHours();
    const int horizonHours = best.size();
    for (int i = 1; i < horizonHours; ++i)
    {
        const int ihour = i + currHour;
        const int hour = ihour % OptimizerEnProfit::HOURS_IN_DAY;
        const int day  =     GMat().Floor(ihour     / static_cast<double>(OptimizerEnProfit::HOURS_IN_DAY));
        const int dayPrev  = GMat().Floor((ihour-1) / static_cast<double>(OptimizerEnProfit::HOURS_IN_DAY));
        if (day != dayPrev)
        {
            //ossInfo << Nl;
        }

        const bool onPrev = best.at(i-1);
        const bool onCurr = best.at(i);
        if (not onPrev && onCurr)
        {
            lastHourOn = hour;
            lastDayOn = day;
        }
        if (onCurr)
        {
            if (firstDayOn < 0)
            {
                firstDayOn = day;
            }
        }
        const bool isInDayLimit = maxDayCmdsLimit < 0 || lastDayOn <= maxDayCmdsLimit;
        const int hourPrev = (ihour - 1) % OptimizerEnProfit::HOURS_IN_DAY;
        if (lastHourOn > 0)
        {
            if (not onCurr) // Switch off
            {
                ossInfo << "day " << firstDayOn << ", hour " << lastHourOn << "-" << hourPrev << Nl;
                if (isInDayLimit)
                {
                    // Wake up
                    ossCmd << "echo \"" << cmdWOL << "\" | at " << lastHourOn << cmdMinuteSuffix << Nl;
                    // Put to sleep
                    ossCmd << cmdsSSH << cmdSuspendAt;
                    if (ihour - 1 < 24)
                    {
                        ossCmd << hourPrev << cmdMinuteSuffix << Nl;
                    }
                    else
                    {
                        ossCmd << hourPrev % 24 << cmdMinuteSuffix << " + " << GMat().Ceil(hourPrev / 24.0) << " days";
                    }
                }

                lastHourOn = -1;
            }
            else if (i == horizonHours - 1)
            {
                ossInfo << "day " << firstDayOn << ", hour " << lastHourOn << "-.." << Nl;
                if (isInDayLimit)
                {
                    // Wake up
                    ossCmd << "echo \"" << cmdWOL << "\" | at " << lastHourOn << cmdMinuteSuffix << Nl;
                }
            }
        }
        if (onCurr && i == 1)
        {
            // Wake up now, right at the beginning! The battery is probably already overloaded.
            ossCmd << cmdWOL << Nl;
            onAtFirstHour = true;
        }
        else
        if (onPrev && not onCurr && isInDayLimit)
        {
            if (onAtFirstHour) // Was started at the beginning already. Be sure to suspend later on.
            {
                ossInfo << "day 0, hour !-" << hourPrev << Nl;
                ossCmd << cmdsSSH << cmdSuspendAt << hourPrev << cmdMinuteSuffix << Nl;
            }
        }
    }
    ossCmd  << Nl;
    ossInfo << Nl;

    ret.commands    = ossCmd.str();
    ret.infos       = ossInfo.str();

    return ret;
}

