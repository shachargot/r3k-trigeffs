import numpy as np
from ROOT import gStyle, gPad, TLegend, TEfficiency, TH1D, TLine, TGraphAsymmErrors
from .PlotBase import PlotBase

class EfficiencyPlot(PlotBase):
    def __init__(self, init_params=None):
        gStyle.SetOptStat(0)
        self.title_string = ';p_{T} [GeV];Efficiency'
        self.x_title = None
        self.y_title = None
        self.color1 = 'black'
        self.color2 = 'blue'
        self.canvas_size = (800,600)
        self.marker_style = None
        self.marker_size = 'small'
        self.xrange = None
        self.yrange = None
        self.rrange = (.5,2)
        self.text_size='med'
        self.leg_pos = 'upper_right'
        self.leg_header = None
        self.leg_scale = None
        self.legtext_size = None
        if init_params: self.set_params(init_params)

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
    
    def eff_ratio(self, eff_1, eff_2):
        hnew1 = eff_1.GetCopyPassedHisto()
        hnew2 = eff_1.GetCopyTotalHisto()
        hnew3 = eff_2.GetCopyPassedHisto()
        hnew4 = eff_2.GetCopyTotalHisto()
        hnew1.Divide(hnew1, hnew2, c1=1., c2=1., option='B')
        hnew3.Divide(hnew3, hnew4, c1=1., c2=1., option='B')
        hnew1.Divide(hnew3)
        return hnew1

    def integrate_eff(self, eff_1, int_floor=2., int_ceil=99999., show=False):
        hnew_pass = eff_1.GetCopyPassedHisto()
        hnew_tot = eff_1.GetCopyTotalHisto()

        num_tot = 0
        den_tot = 0
        nbins = hnew_pass.GetNbinsX() - 1
        for i in range(nbins):
            if hnew_pass.GetBinLowEdge(i) < int_floor or hnew_pass.GetBinLowEdge(i) > int_ceil: continue
            num_tot += hnew_pass.GetBinContent(i)
            den_tot += hnew_tot.GetBinContent(i)

        eff = num_tot/den_tot if den_tot else print(num_tot)
        eff = round(eff,3) if num_tot and den_tot else 0.
        err = 0. if eff==0. else '{:0.2e}'.format(np.sqrt((eff*(1-eff))/den_tot),5)
        if show: print(f'Integrated Eff = {num_tot} / {den_tot} = {eff} \pm {err}')
        return eff, err

    # Core function for plot generation
    def plotEfficiencies(self, h1, h2, ratio=True, h1_title=None, h2_title=None, show=False, save=False, addIntegral=False):
        # Construct plot objects
        h1_pass = h1.GetCopyPassedHisto()
        h1_tot = h1.GetCopyTotalHisto()
        h2_pass = h2.GetCopyPassedHisto()
        h2_tot = h2.GetCopyTotalHisto()

        eff1 = TEfficiency(h1_pass, h1_tot)
        eff2 = TEfficiency(h2_pass, h2_tot)
        self.format_entry(eff1, title=h1_title, line_color=self.color1, marker_style='' if self.marker_style is None else self.marker_style, marker_color=self.color1, marker_size=self.marker_size)
        self.format_entry(eff2, title=h2_title, line_color=self.color2, marker_style='' if self.marker_style is None else self.marker_style, marker_color=self.color2, marker_size=self.marker_size)
        self.yrange = (0.,1.) if self.yrange is None else self.yrange

        # Legend object
        leg = TLegend(0, 0, .5, .5)
        entry1 = f'{eff1.GetTitle()}'
        entry2 = f'{eff2.GetTitle()}'

        # Efficiency integral
        if addIntegral:
            if isinstance(addIntegral, bool):
                int_floor = h1_pass.GetBinLowEdge(1)
                int_ceil = h1_pass.GetBinLowEdge(h1_pass.GetNbinsX())
            else:
                int_floor, int_ceil = addIntegral
            eff1_int = self.integrate_eff(eff1, int_floor=int_floor, int_ceil=int_ceil)
            eff2_int = self.integrate_eff(eff2, int_floor=int_floor, int_ceil=int_ceil)
            entry1 = entry1+f' \\ [ \epsilon = {eff1_int[0]} \pm {eff1_int[1]}]'
            entry2 = entry2+f' \\ [ \epsilon = {eff2_int[0]} \pm {eff2_int[1]}]'

        if self.leg_header: leg.SetHeader(self.leg_header,'C')
        leg.AddEntry(eff1, entry1, 'le' if self.marker_style is None else 'pl')
        leg.AddEntry(eff2, entry2, 'le' if self.marker_style is None else 'pl')

        # Include ratio panel
        if ratio:
            c, p1, p2 = self.createCanvas(option='ratio', size=self.canvas_size)

            ## Top Panel
            p1.cd()

            # Primary plot
            frame = eff1.GetTotalHistogram().Clone("frame")
            frame.Reset()
            frame.SetTitle(self.title_string)
            frame.SetMinimum(self.yrange[0])
            frame.SetMaximum(self.yrange[1])
            frame.SetStats(0)
            frame.Draw("AXIS")

            eff1.Draw("P SAME")
            eff2.Draw("P SAME E")

            self.format_axes( frame,
                              option='upper',
                              xrange=self.xrange,
                              yrange=self.yrange,
                              text_size=self.text_size,
                              title_string=self.title_string,
                              x_title=self.x_title,
                              y_title=self.y_title,)

            # Legend
            leg.Draw()
            self.format_legend(leg, pos=self.leg_pos, option='upper', scale=self.leg_scale, legtext_size=self.legtext_size)

            ## Ratio Panel
            gPad.Update()
            p2.cd()

            # Efficiency ratio
            r = self.eff_ratio(eff1, eff2)
            r.Draw()
            self.format_entry(r)
            _ = self.format_axes(r, option='lower', xrange=self.xrange, yrange=self.rrange, text_size=self.text_size)

            # TODO Ratio legend 

            gPad.Update()
            c.cd()
            gPad.Update()
            c.Update()

            if show: c.Draw()
            if save: c.SaveAs(save)
            return c, r

        # No ratio panel
        else:
            c = self.createCanvas(option='hist', size=self.canvas_size)
            c.cd()
            # Primary plot
            eff1.Draw()
            self.format_axes(eff1, option='full', xrange=self.xrange, yrange=self.yrange, text_size=self.text_size)
            eff2.Draw('SAME')

            # Legend
            self.format_legend(leg, option='full', pos=self.leg_pos)
            leg.Draw()

            gPad.Update()
            c.Update()
        
        if save: c.SaveAs(save)

    def plotEfficiencyGraphs(self, g_data, g_mc, g_sf, bins, h1_title=None, h2_title=None,
                         show=False, save=False, ratio_line=True):
        """Efficiency plot from pre-computed graphs (e.g. from a fit) instead of pass/total histograms.

        g_data, g_mc : TGraphAsymmErrors of efficiency vs. bin centre (asymmetric y errors allowed)
        g_sf         : TGraphAsymmErrors of the scale factor (data/MC), drawn in the ratio pad
        bins         : bin edges (array-like), used for the axis frame
        """
        from array import array
        from ROOT import kBlack, kBlue

        edges = array('d', [float(b) for b in bins])
        nb = len(edges) - 1

        def style(g, color, marker, title):
            g.SetTitle(title or '')
            g.SetLineColor(color)
            g.SetMarkerColor(color)
            g.SetMarkerStyle(self.marker_style if isinstance(self.marker_style, int) else marker)
            g.SetMarkerSize(0.8)
            g.SetLineWidth(2)

        style(g_data, kBlack, 20, h1_title)
        style(g_mc, kBlue, 21, h2_title)
        style(g_sf, kBlack, 20, 'Data / MC')

        # Legend
        leg = TLegend(0, 0, .5, .5)
        if self.leg_header:
            leg.SetHeader(self.leg_header, 'C')
        leg.AddEntry(g_data, h1_title or 'Data', 'pe')
        leg.AddEntry(g_mc, h2_title or 'MC', 'pe')

        c, p1, p2 = self.createCanvas(option='ratio', size=self.canvas_size)

        # ---- Top panel
        p1.cd()
        frame = TH1D('frame_eff', self.title_string, nb, edges)
        frame.SetDirectory(0)
        frame.SetStats(0)
        frame.SetMinimum(self.yrange[0] if self.yrange else 0.)
        frame.SetMaximum(self.yrange[1] if self.yrange else 1.1)
        frame.Draw('AXIS')

        if g_mc.GetN():
            g_mc.Draw('P SAME')
        if g_data.GetN():
            g_data.Draw('P SAME')

        self.format_axes(frame,
                         option='upper',
                         xrange=self.xrange,
                         yrange=self.yrange,
                         text_size=self.text_size,
                         title_string=self.title_string,
                         x_title=self.x_title,
                         y_title=self.y_title)

        leg.Draw()
        self.format_legend(leg, pos=self.leg_pos, option='upper', scale=self.leg_scale,
                           legtext_size=self.legtext_size)

        # ---- Ratio panel
        gPad.Update()
        p2.cd()
        rframe = TH1D('frame_ratio', ';;Data / MC', nb, edges)
        rframe.SetDirectory(0)
        rframe.SetStats(0)
        rframe.SetMinimum(self.rrange[0])
        rframe.SetMaximum(self.rrange[1])
        rframe.Draw('AXIS')
        self.format_axes(rframe, option='lower', xrange=self.xrange, yrange=self.rrange,
                         text_size=self.text_size)

        line = None
        if ratio_line:
            line = TLine(edges[0], 1.0, edges[-1], 1.0)
            line.SetLineStyle(7)
            line.Draw()
        if g_sf.GetN():
            g_sf.Draw('P SAME')

        gPad.Update()
        c.cd()
        gPad.Update()
        c.Update()

        # keep ROOT objects alive as long as the plot object is
        self._keep = [frame, rframe, line, leg, g_data, g_mc, g_sf]

        if show:
            c.Draw()
        if save:
            c.SaveAs(save)
        return c, g_sf
