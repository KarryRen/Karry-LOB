# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 2024/02/28 11:25
#
# pylint: disable=no-member

""" The High-Frequency Trading Label (HFTLabel) algorithm. """

import numpy as np

from ..tensor_construction_base import ConstructionAlgoBase
from Ops import CodeType, Status, Code


class HFTLabel(ConstructionAlgoBase):
    """ The High-Frequency Trading Label(HFTLabel) algorithm. """

    def __init__(self, code_type: CodeType = CodeType["UNKNOWN"], **kwargs):
        """ Init function of HFTLabel.

        :param code_type: the type of code

        """

        # ---- Define the step of label (K) ---- #
        if code_type == CodeType["ETF"]:
            self.label_step = int(60)
        else:
            self.label_step = int(10)

        # ---- Define the special params for HFTLabel ---- #
        version = (2024, 3, 25, 17, 0)  # use date to note version
        out_coords = {
            "T": list(range(28800)),
            "F": [f"label_ret_{self.label_step}_way1", f"label_ret_{self.label_step}_way3",
                  f"label_diff_{self.label_step}_way1", f"label_diff_{self.label_step}_way3",
                  f"labelweight_ret_{self.label_step}_way1", f"labelweight_ret_{self.label_step}_way3",
                  f"labelweight_diff_{self.label_step}_way1", f"labelweight_diff_{self.label_step}_way3"]
        }  # the out coords
        init_data = 0.0  # the default init data
        data_source = {
            "LOB": {"features": ["Price"], "levels": 1},
            "Trade": {"features": ["Status"]},
        }  # the data source

        # ---- Init the base Class ---- #
        super(HFTLabel, self).__init__(version=version, out_coords=out_coords, data_source=data_source, init_data=init_data, **kwargs)

    def cal_fea(self, code: Code, date: str):
        """ HTFLabel calculation algorithm. """

        # ---- Step 1. Get the market data of the `code` in `date` based on the data_source ---- #
        # ds_array_well is the flag of reading data_source right or not
        # ds_xray_dict is the dict of each data_source type and xarray
        ds_array_well, ds_xray_dict = self.get_all_market_data(code, date)

        # ---- Step 2. Compute the labels based on the ds_xray ---- #
        if np.all(ds_array_well):  # all datasource array well
            # Read the data
            xray_lob = ds_xray_dict["LOB"]  # the LOB data
            xray_trade = ds_xray_dict["Trade"]  # the Trade data
            epsilon = 1e-5  # the epsilon is used for avoiding 0 denominator and used to judge float num == 0

            # Compute the Status using to judge tick is trading or not
            xray_trade = np.abs(xray_trade.loc[:, "Status"] - Status["TRADING"].value) < epsilon

            # Compute the Price
            xray_bidprice = xray_lob.loc[:, "Bid", 1, "Price"]
            xray_askprice = xray_lob.loc[:, "Ask", 1, "Price"]
            xray_bidprice_weight = (xray_bidprice > epsilon).astype(float)  # judge == 0(weight = 0) or not(weight = 1)
            xray_askprice_weight = (xray_askprice > epsilon).astype(float)  # judge == 0(weight = 0) or not(weight = 1)
            xray_midprice = (xray_bidprice * xray_bidprice_weight + xray_askprice * xray_askprice_weight) / (xray_bidprice_weight + xray_askprice_weight)
            assert (np.sum(np.isnan(xray_midprice.data[xray_trade.data])) == 0), "Mid Price ERROR !!! In Trading tick, Mid Price is NAN !!!"

            # Gen label using 2 different ways => gen `k` Nan
            array_price = xray_midprice.data  # shape=(tick_num,) - 1 dim array
            array_price_valid = xray_trade.data  # shape=(tick_num,) - 1 dim array
            for label_way in ["way1", "way3"]:  # 2 label ways
                # define the empty label array
                array_label = np.zeros_like(array_price)
                array_labelweight = np.zeros_like(array_price)
                # compute the future price based on different way
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
                assert (np.min(array_price[both_price_valid_index]) > epsilon), "Now array_price Wrong !!!"
                assert (np.min(array_future_price_valid[both_price_valid_index]) > epsilon), "Future array_price Wrong !!!"
                # set the label weight
                array_labelweight[both_price_valid_index] = 1.0
                # calculate the label value
                for cal_way in ["ret", "diff"]:  # two different calculating ways
                    # set the weight to different kind of labels (all same)
                    labelweight_name = f"labelweight_{cal_way}_{self.label_step}_{label_way}"
                    self.xray.loc[:, labelweight_name].data[:] = array_labelweight  # the label weights are all same
                    # compute the label value (different way has different computing function)
                    label_name = f"label_{cal_way}_{self.label_step}_{label_way}"
                    if cal_way == "ret":
                        array_label[both_price_valid_index] = (np.log(array_future_price[both_price_valid_index]) - np.log(
                            array_price[both_price_valid_index])) * 10000
                    elif cal_way == "diff":
                        array_label[both_price_valid_index] = array_future_price[both_price_valid_index] - array_price[both_price_valid_index]
                    self.xray.loc[:, label_name].data[:] = array_label
        return self.xray
