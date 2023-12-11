from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_kandinsky_2_image_mixing.infer_kandinsky_2_image_mixing_process import InferKandinsky2ImageMixingFactory
        return InferKandinsky2ImageMixingFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_kandinsky_2_image_mixing.infer_kandinsky_2_image_mixing_widget import InferKandinsky2ImageMixingWidgetFactory
        return InferKandinsky2ImageMixingWidgetFactory()
