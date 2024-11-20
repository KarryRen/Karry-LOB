# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/02/28 11:25

""" The High-Frequency Trading Label (HFTLabel) algorithm. """

import numpy as np

from ..tensor_construction_base import ConstructionAlgoBase
from Ops import CodeType, Status, Code


class HFTLabel(ConstructionAlgoBase):
    """ The High-Frequency Trading Label(HFTLabel) algorithm. """

    def __init__(self, **kwargs):
        """ Init function of HFTLabel.

        :param code_type: the type of code

        """

        # ---- Define the step of label (k) ---- #
        self.label_step = int(10)

        # ---- Define the special params for HFTLabel ---- #
        version = (2024, 11, 20, 14, 51)  # use date to note version
        out_coords = {
            "T": list(range(28800)),
            "F": [
                f"label_ret_{self.label_step}_way1", f"label_ret_{self.label_step}_way3",
                f"label_weight_ret_{self.label_step}_way1", f"label_weight_ret_{self.label_step}_way3"
            ]
        }  # the out coords, there are 2 label ways
        init_data = 0.0  # the default init data
        data_source = {"LOB": {"features": ["Price"], "levels": 1}, "Trade": {"features": ["Status"]}}  # the data source

        # ---- Init the base Class ---- #
        super(HFTLabel, self).__init__(version=version, out_coords=out_coords, data_source=data_source, init_data=init_data, **kwargs)

    def cal_fea(self, code: Code, date: str):
        """ HTFLabel calculation algorithm. """

        # ---- Get the market data of the `code` in `date` based on the data_source ---- #
        # ds_array_well is the flag of reading data_source right or not
        # ds_xray_dict is the dict of each data_source type and xarray
        ds_array_well, ds_xray_dict = self.get_all_market_data(code, date)

        # ---- Compute the labels based on the ds_xray ---- #
        if np.all(ds_array_well):  # all datasource array well
            # read the data
            xray_lob = ds_xray_dict["LOB"]  # the LOB data
            xray_trade = ds_xray_dict["Trade"]  # the Trade data
            epsilon = 1e-5  # the epsilon is used for judging `equal`

            # compute the status using to judge tick is trading or not. equal means TRADING.
            xray_trade = np.abs(xray_trade.loc[:, "Status"] - Status["TRADING"].value) < epsilon

            # get the price & compute the midprice
            xray_bidprice, xray_askprice = xray_lob.loc[:, "Bid", 1, "Price"], xray_lob.loc[:, "Ask", 1, "Price"]
            xray_bidprice_weight, xray_askprice_weight = (xray_bidprice > epsilon).astype(float), (xray_askprice > epsilon).astype(float)
            xray_midprice = (xray_bidprice * xray_bidprice_weight + xray_askprice * xray_askprice_weight) / (
                    xray_bidprice_weight + xray_askprice_weight)
            assert np.sum(np.isnan(xray_midprice.data[xray_trade.data])) == 0, "ERROR: Mid Price ERROR !!! In Trading tick, Mid Price is NAN !!!"

            # gen label using 2 different ways
            array_price = xray_midprice.data  # shape=(tick_num,)
            array_price_valid = xray_trade.data  # shape=(tick_num,)
            for label_way in ["way1", "way3"]:
                # define the empty label array
                array_label, array_label_weight = np.zeros_like(array_price), np.zeros_like(array_price)
                # compute the `future` price based on different way => gen `k` Nan
                if label_way == "way1":
                    array_future_price = xray_midprice.rolling(timestamp=self.label_step).mean().shift(timestamp=-self.label_step).fillna(0.0).data
                    array_future_price_valid = xray_trade.rolling(timestamp=self.label_step).min().shift(timestamp=-self.label_step).fillna(0.0).data
                elif label_way == "way3":
                    array_future_price = xray_midprice.shift(timestamp=-self.label_step).fillna(0.0).data
                    array_future_price_valid = xray_trade.shift(timestamp=-self.label_step).fillna(0.0).data
                else:
                    raise ValueError(label_way)
                # get both now and future price valid index and test both the now and future prices are OK
                both_price_valid_index = (array_price_valid > (1 - epsilon)) & (array_future_price_valid > (1 - epsilon))
                assert (np.min(array_price[both_price_valid_index]) > epsilon), "ERROR: Now array_price Wrong !!!"
                assert (np.min(array_future_price_valid[both_price_valid_index]) > epsilon), "ERROR: Future array_price Wrong !!!"
                # set the label weight
                array_label_weight[both_price_valid_index] = 1.0
                label_weight_name = f"label_weight_ret_{self.label_step}_{label_way}"
                self.xray.loc[:, label_weight_name].data[:] = array_label_weight
                # set the label
                label_name = f"label_ret_{self.label_step}_{label_way}"
                array_label[both_price_valid_index] = np.log(array_future_price[both_price_valid_index] / array_price[both_price_valid_index]) * 10000
                self.xray.loc[:, label_name].data[:] = array_label
        return self.xray
